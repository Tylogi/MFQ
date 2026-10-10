#include "../runtime/execution_options.h"
#include "moe_ffn_pipeline.h"
#include "moe_cached_source_internal.h"
#include "cpu_projection_rows.h"
#include "mfq/cpu_expert_pool.h"
#include "runtime/graph_registry.h"
#include "runtime/moe_pipeline.h"
#include "mfq_cuda_moe_ops.h"
#include "runtime/decode_window.h"
#include "runtime/dma_copy_batch.h"
#include "mfq_cuda_context.h"
#include "mfe_ffn_runtime.h"
#include "quant_linear.h"
#include <cuda.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <map>
#include <sstream>
#include <stdexcept>

namespace tb=mfq_tensor_backend;
namespace {
using StreamFlagWrite = decltype(&cuStreamWriteValue32);
void* stream_flag_driver_entry(const char* symbol) {
#if CUDA_VERSION >= 12000 && CUDART_VERSION >= 12000
    void* pointer=nullptr;
    cudaDriverEntryPointQueryResult result=cudaDriverEntryPointSymbolNotFound;
    const auto error=cudaGetDriverEntryPoint(symbol,&pointer,cudaEnableDefault,&result);
    if(error==cudaErrorNotSupported) {cudaGetLastError();return nullptr;}
    MFQ_CUDA_CHECK(error);
    return result==cudaDriverEntryPointSuccess?pointer:nullptr;
#else
    (void)symbol;return nullptr;
#endif
}
StreamFlagWrite stream_flag_writer() {
#if CUDA_VERSION >= 12000 && CUDART_VERSION >= 12000
    static const auto write=reinterpret_cast<StreamFlagWrite>(stream_flag_driver_entry("cuStreamWriteValue32"));
    return write;
#else
    return nullptr;
#endif
}

// Mapped plans/flags have fixed addresses for every replay. Counting their
// allocations supplements the process's physical-memory measurement.
struct Mapped {
    void* host=nullptr;void* device=nullptr;std::size_t bytes=0;
    std::shared_ptr<Mapped> shared_owner;
    explicit Mapped(std::size_t n):bytes(n) {
        MFQ_CUDA_CHECK(cudaHostAlloc(&host,n,cudaHostAllocMapped));
        const auto status=cudaHostGetDevicePointer(&device,host,0);
        if(status!=cudaSuccess) {cudaFreeHost(host);host=nullptr;MFQ_CUDA_CHECK(status);}
        mfq::cuda::charge_tensor_host_bytes(n);std::memset(host,0,n);
    }
    Mapped(const Mapped&)=delete;
    Mapped& operator=(const Mapped&)=delete;
    void borrow(std::shared_ptr<Mapped> owner) {
        if(host && !shared_owner){cudaFreeHost(host);mfq::cuda::tensor_host_bytes.fetch_sub(bytes);}
        shared_owner=std::move(owner);host=shared_owner->host;device=shared_owner->device;bytes=shared_owner->bytes;
    }
    ~Mapped(){if(host && !shared_owner){cudaFreeHost(host);mfq::cuda::tensor_host_bytes.fetch_sub(bytes);}}
    template<class T>T* h()const{return static_cast<T*>(host);}
    template<class T>T* d()const{return static_cast<T*>(device);}
};
void publish(uint32_t* flag) {
    std::atomic_thread_fence(std::memory_order_seq_cst);_mm_sfence();*(volatile uint32_t*)flag=1;
}
MixedMoePool fields_pool(MixedMoePool pool,const std::vector<tb::Tensor>& fields,int slots,int output) {
    pool.local_experts=slots;
    if(pool.family==MixedMoeFamily::Nint) {
        auto& w=pool.nint;w.workspaces.clear();w.aligned_q8=false;
        w.q_packed=fields[0];w.row_q_bits=fields[1];w.row_q_bit_offsets=fields[2];
        w.sub_scale=fields[3];w.sub_min=fields[4];w.neuron_scale=fields[5];w.neuron_min=fields[6];
        w.out=int64_t(slots)*output;if(!w.shape.empty())w.shape[0]=w.out;
    } else if(pool.family==MixedMoeFamily::Nvq) {
        auto& w=pool.nvq;w.workspaces.clear();w.indices_packed=fields[0];w.aux_packed=fields[1];
        w.sub_scale_packed=fields[2];w.neuron_scale=fields[3];
        w.out=int64_t(slots)*output;if(!w.shape.empty())w.shape[0]=w.out;
    } else throw std::runtime_error("MFQ FFN CPU callback lacks this packed format");
    return pool;
}
CpuProjectionRows rows_plan(const MixedMoePool& p) {
    return p.family==MixedMoeFamily::Nint ? make_cpu_projection_rows(p.nint) : make_cpu_projection_rows(p.nvq);
}

std::function<mfq::MoeCpuCalibration::Observation()> cpu_calibration_work(
        mfq::MoeCpuCostModel::Key key,std::array<MoeHostExpertCache::Lease,3> leases,
        std::vector<float> input,int width,int intermediate,int count,double preparation_ns) {
    return [key,leases=std::move(leases),input=std::move(input),width,intermediate,count,preparation_ns]() {
        const auto started=std::chrono::steady_clock::now();
        const std::array<CpuProjectionRows,3> rows{
            rows_plan(leases[0]->weights),rows_plan(leases[1]->weights),rows_plan(leases[2]->weights)};
        std::vector<float> gate(size_t(count)*intermediate),up(gate.size()),hidden(gate.size()),
            output(size_t(count)*width);
        mfq::cpu::expert_pool().rows(intermediate,[&](int64_t begin,int64_t end) {
            rows[0].run(input.data(),count,width,gate.data(),intermediate,int(begin),int(end));
            rows[1].run(input.data(),count,width,up.data(),intermediate,int(begin),int(end));
            for(int row=0;row<count;++row)for(int column=int(begin);column<int(end);++column) {
                const auto i=size_t(row)*intermediate+column;
                const mfq_half g(gate[i]),u(up[i]);
                const mfq_half sigmoid(1.0/(1.0+std::exp(-double(float(g)))));
                const mfq_half first(float(g)*float(sigmoid));
                hidden[i]=float(mfq_half(float(first)*float(u)));
            }
        });
        mfq::cpu::expert_pool().rows(width,[&](int64_t begin,int64_t end) {
            rows[2].run(hidden.data(),count,intermediate,output.data(),width,int(begin),int(end));
        });
        const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count();
        return mfq::MoeCpuCalibration::Observation{key,preparation_ns+double(ns)};
    };
}

}

struct MoeFfnPrefillHostWorkspace {
    std::shared_ptr<Mapped> x,ids,cpu;
    MoeFfnPrefillHostWorkspace(int tokens,int routes,int width)
        :x(std::make_shared<Mapped>(std::size_t(tokens)*width*sizeof(float))),
         ids(std::make_shared<Mapped>(std::size_t(tokens)*routes*sizeof(int32_t))),
         cpu(std::make_shared<Mapped>(std::size_t(tokens)*routes*width*sizeof(mfq_half))) {}
};

class MoeFfnPipeline : public std::enable_shared_from_this<MoeFfnPipeline> {
    struct CopyEvents {
        cudaEvent_t fork=nullptr,done=nullptr;
        CopyEvents() {
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&fork,cudaEventDisableTiming));
            const auto status=cudaEventCreateWithFlags(&done,cudaEventDisableTiming);
            if(status!=cudaSuccess){cudaEventDestroy(fork);fork=nullptr;MFQ_CUDA_CHECK(status);}
        }
        ~CopyEvents(){if(done)cudaEventDestroy(done);if(fork)cudaEventDestroy(fork);}
    };
    struct PhaseTimes {
        std::array<cudaEvent_t,4> points{};
        void initialize() {
            if(points[0])return;
            try {for(auto& event:points)MFQ_CUDA_CHECK(cudaEventCreate(&event));}
            catch(...) {for(auto& event:points)if(event){cudaEventDestroy(event);event=nullptr;}throw;}
        }
        ~PhaseTimes(){for(auto event:points)if(event)cudaEventDestroy(event);}
    };
    struct DmaTimes {
        std::array<cudaEvent_t,3> points{};
        MoeExpertCache::DmaCallSample sample;
        std::chrono::steady_clock::time_point fetch_started{},enqueue_started{};
        bool started=false;
        DmaTimes() {
            try {
                for(auto& event:points) {
                    const auto status=cudaEventCreate(&event);
                    if(status==cudaErrorMemoryAllocation)throw std::bad_alloc();
                    MFQ_CUDA_CHECK(status);
                }
            } catch(...) {for(auto event:points)if(event)cudaEventDestroy(event);throw;}
        }
        ~DmaTimes(){for(auto event:points)if(event)cudaEventDestroy(event);}
    };
    struct Binding {
        MixedMoeRuntime hot,transfer;
        std::vector<MoeGpuArena*> stages;
        std::vector<tb::Tensor> owner;
    };
    struct CpuJob {
        int expert=0,offset=0,count=0;
        std::array<MoeHostExpertCache::Lease,3> leases;
        std::array<CpuProjectionRows,3> rows;
    };
    struct TransferTemplate {
        mfq::cuda::MfePackedProjection view;
        std::array<const uint8_t*,7> bases{};
    };
    struct Cell {
        int tokens,routes,entries;
        int calibration_expert=-1;
        tb::Tensor input,ids,weights,output,dispatch_pointers,wire,device_abort,device_kind;
        tb::Tensor shared_input_descriptors;
        int shared_input_groups=0;
        bool shared_input=false;
        bool gpu_resident_dispatch=false;
        bool fused_activation=false;
        bool two_stage=false;
        bool graph_replay=true;
        std::unique_ptr<mfq::cuda::Event> eager_done;
        bool skip_cpu_wait=false;
        bool skip_gpu_timing=false;
        bool window_input_views=false;
        bool transfer_cache=false;
        bool mapped_copy=false;
        bool mapped_overlap=false;
        bool phased_transfer=false;
        bool reuse_plan=false;
        bool check_reused_views=false;
        std::array<MixedMoeRuntime,3> fused_projections;
        std::unique_ptr<MfeFfnRuntime> fused;
        std::unique_ptr<Mapped> transfer_index,transfer_descriptors;
        std::unique_ptr<Mapped> mapped_copy_table;
        tb::Tensor mapped_copy_snapshot;
        std::unique_ptr<CopyEvents> copy_events;
        std::vector<mfq::cuda::MfePackedProjection> transfer_views;
        std::array<std::vector<TransferTemplate>,3> transfer_templates;
        tb::Tensor shared_gate;
        int64_t wire_header_bytes=0;
        int wire_blocks=0,wire_threads=0;
        Mapped x,host_ids,flags,kind,local,cpu;
        std::array<Binding,3> binding;
        PhaseTimes timing;
        std::unique_ptr<DmaTimes> dma_timing;
        std::unique_ptr<mfq::cuda::GraphRegistry> graphs;
        std::vector<float> cpu_input,g,u,hidden,y;
        std::vector<CpuJob> jobs;
        std::uint64_t cpu_profile_calls=0;
        mfq::MoeDispatchPlan plan;
        mfq::MoeDispatchWorkspace plan_workspace;
        std::vector<int32_t> plan_ids;
        bool pending=false,in_window=false;
        StreamFlagWrite ready_writer=nullptr;
        Cell(int t,int k,int width,int experts,cudaStream_t stream,bool eager):tokens(t),routes(k),entries(t*k),
            x(eager?64:std::size_t(t)*width*sizeof(float)),host_ids(eager?64:std::size_t(t)*k*sizeof(int32_t)),flags(6*64),
            kind(std::size_t(experts)*sizeof(int32_t)),local(std::size_t(experts)*3*sizeof(int32_t)),
            cpu(eager?64:std::size_t(t)*k*width*sizeof(mfq_half)),graphs(std::make_unique<mfq::cuda::GraphRegistry>(stream)) {
            const auto gpu=tb::TensorOptions().device(tb::kCUDA);
            input=tb::zeros({t,width},gpu.dtype(tb::kFloat16));ids=tb::zeros({t,k},gpu.dtype(tb::kInt32));
            weights=tb::zeros({t,k},gpu.dtype(tb::kFloat32));
        }
    };
    std::array<std::shared_ptr<MoeCachedSource>,3> sources_;
    MoeExpertCache* cache_;
    MoeFfnShared shared_;
    MoeFfnSharedWeights shared_weights_;
    MoeFfnDispatchObserver observer_;
    std::map<std::tuple<int,int,const void*>,std::unique_ptr<Cell>> cells_;
    struct PredictionCell {
        Mapped flags{128};
        tb::Tensor aborted=tb::zeros({1},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
        PredictionCell(){publish(flags.h<uint32_t>()+4);publish(flags.h<uint32_t>()+8);}
    };
    std::map<const void*,std::shared_ptr<PredictionCell>> prediction_cells_;
    int width_,ff_,experts_;
    cudaStream_t stream_=nullptr;
    bool failed_=false;
    bool trace_serve_=false;
    double cpu_ns_=0,cpu_work_=0,gpu_ns_=0,gpu_samples_=0;
    double stage_ns_=0,stage_bytes_=0;
    std::vector<mfq::MoeCpuCostModel::Key> cpu_cost_keys_;
    mfq::MoeCpuCostModel::Key cpu_cost_key(const mfq::MoeDispatchGroup& group)const {
        auto key=cpu_cost_keys_[group.expert];key[3]=group.positions.size();return key;
    }

    #include "moe_ffn_prefill_stream.inc"
    #include "moe_ffn_prefill_layer.inc"

    MoeGpuArena& stage(int projection,MoeCachedCohort& cohort) {
        const auto key=std::to_string(projection)+":"+cohort.arena->signature;
        auto& arena=cache_->pipeline_stages_[key];
        if(arena)return *arena;
        arena=std::make_unique<MoeGpuArena>();arena->signature=key;arena->slot_bytes=cohort.arena->slot_bytes;
        arena->layouts=cohort.arena->layouts;
        // Full source geometry bounds the number of distinct experts in a
        // cohort, independent of token batching; addresses never grow later.
        for(const auto& source:cache_->sources_)for(const auto& c:source->cohorts_)
            if(c.arena->signature==cohort.arena->signature)arena->slots=std::max(arena->slots,c.cpu->local_experts);
        for(const auto& layout:arena->layouts) {
            auto shape=layout.slot_shape;shape[0]*=arena->slots;
            arena->fields.push_back(tb::empty(shape,tb::TensorOptions().device(tb::kCUDA).dtype(layout.scalar_type)));
        }
        return *arena;
    }
    void initialize(Cell& cell,CudaExecutionContext& execution,bool window=false) {
        cell.in_window=window;
        const auto* prefill_graph=std::getenv("MFQ_MOE_PREFILL_GRAPH");
        cell.graph_replay=window || cell.tokens<=8 || !prefill_graph || std::atoi(prefill_graph)!=0;
        if(!cell.graph_replay) {
            cell.eager_done=std::make_unique<mfq::cuda::Event>(cudaEventDisableTiming);
            const auto key=std::to_string(cell.tokens)+":"+std::to_string(cell.routes)+":"+std::to_string(width_);
            auto& workspace=cache_->pipeline_prefill_host_workspaces_[key];
            if(!workspace)workspace=std::make_shared<MoeFfnPrefillHostWorkspace>(cell.tokens,cell.routes,width_);
            // The next route publication follows the previous merge on the
            // compute stream. Host input reads and CPU writes start only after
            // that publication, so layers cannot use this scratch concurrently.
            cell.x.borrow(workspace->x);cell.host_ids.borrow(workspace->ids);cell.cpu.borrow(workspace->cpu);
        }
        cell.ready_writer=window?stream_flag_writer():nullptr;
        cell.two_stage=execution.config.moe_two_stage_ffn && cell.tokens<=6 && cell.routes+(shared_?1:0)<=32 &&
            execution.kl_mmq.mode==KlMmqMode::Default;
        for(const auto& source:sources_)for(const auto& cohort:source->cohorts_)
            cell.two_stage=cell.two_stage && (cohort.active.family==MixedMoeFamily::Nint || cohort.active.family==MixedMoeFamily::Nvq);
        if(shared_) {
            cell.two_stage=cell.two_stage && bool(shared_weights_.gate);
            for(const auto& weight:shared_weights_.projections)
                cell.two_stage=cell.two_stage && weight && weight->is_nint() && !weight->tensor_parallel() && !weight->nint.q8_zero;
        }
        const auto* reuse=std::getenv("MFQ_MOE_PLAN_REUSE");
        cell.reuse_plan=cell.two_stage && (!reuse || std::atoi(reuse)!=0);
        const auto* check=std::getenv("MFQ_MOE_PLAN_REUSE_CHECK");
        cell.check_reused_views=cell.reuse_plan && check && std::atoi(check)!=0;
        const auto* trace=std::getenv("MFQ_TRACE_MFE_DISPATCH");
        if(trace && std::atoi(trace)!=0)
            std::fprintf(stderr,"moe_plan_reuse_runtime tokens=%d routes=%d reuse=%d check=%d\n",
                cell.tokens,cell.routes,int(cell.reuse_plan),int(cell.check_reused_views));
        // A fixed full PCIe quota schedules every cold expert on the GPU.
        // CPU dispatch replays retain their original wait, even with that quota.
        cell.skip_cpu_wait=cell.two_stage && mfq::cuda::runtime_options::skip_gpu_only_cpu_wait() &&
            cache_->config_.moe_ram_pcie_fraction && cache_->ram_pcie_fraction_==1.0 &&
            std::all_of(cache_->pipeline_dispatch_replay_.begin(),cache_->pipeline_dispatch_replay_.end(),
                [](const auto& record){return record.cpu.empty();});
        // With a fixed GPU quota no dispatch decision consumes the measured
        // arithmetic rate. Keep timing whenever detailed DMA profiling needs
        // these event origins, or a dispatch replay can contain CPU work.
        cell.skip_gpu_timing=cell.two_stage && mfq::cuda::runtime_options::skip_fixed_gpu_timing() &&
            cache_->config_.moe_ram_pcie_fraction && cache_->ram_pcie_fraction_==1.0 &&
            !mfq::cuda::runtime_options::moe_diagnostics().dma &&
            std::all_of(cache_->pipeline_dispatch_replay_.begin(),cache_->pipeline_dispatch_replay_.end(),
                [](const auto& record){return record.cpu.empty();});
        if(!cell.skip_gpu_timing)cell.timing.initialize();
        cell.gpu_resident_dispatch=execution.config.moe_gpu_resident_dispatch && cell.tokens<=8;
        cell.fused_activation=execution.config.moe_ffn_fused_activation;
        if((cell.fused_activation || cell.two_stage) && !cache_->pipeline_sigmoid_table_.defined())
            cache_->pipeline_sigmoid_table_=moe_swiglu_sigmoid_table_cuda();
        if(cell.two_stage) {
            std::array<std::vector<int32_t>,3> mapping;
            for(int p=0;p<3;++p) {
                auto& runtime=cell.fused_projections[p];const auto& source=*sources_[p];
                runtime.n_experts=experts_;runtime.out_per_expert=source.cpu_->out_per_expert;
                runtime.neuron_len=source.cpu_->neuron_len;runtime.partial_experts=true;
                for(const auto& cohort:source.cohorts_)runtime.pools.push_back(cohort.active);
                mapping[p]=source.expert_to_cohort_;
            }
            std::array<const NintWeight*,3> shared{};
            if(shared_) {
                for(int p=0;p<3;++p)shared[p]=&shared_weights_.projections[p]->nint;
                cell.shared_gate=shared_weights_.gate(execution,cell.input).contiguous();
            }
            cell.fused=std::make_unique<MfeFfnRuntime>(std::array<MixedMoeRuntime*,3>{&cell.fused_projections[0],
                &cell.fused_projections[1],&cell.fused_projections[2]},cell.input,cell.ids,cell.weights,
                cache_->pipeline_sigmoid_table_,shared,cell.shared_gate,width_,ff_,&mapping,!shared_);
            cell.transfer_index=std::make_unique<Mapped>(std::size_t(experts_)*sizeof(int32_t));
            std::memset(cell.transfer_index->host,0xff,cell.transfer_index->bytes);
            cell.transfer_descriptors=std::make_unique<Mapped>(std::size_t(cell.entries)*3*sizeof(mfq::cuda::MfePackedProjection));
            cell.fused->asynchronous(cell.kind.d<int32_t>(),cell.flags.d<uint32_t>()+16,cell.flags.d<uint32_t>()+32,
                cell.flags.d<uint32_t>()+48,cell.flags.d<uint32_t>()+64,cell.transfer_index->d<int32_t>(),cell.cpu.device);
            if(cell.reuse_plan)for(int p=0;p<3;++p) {
                const auto& source=*sources_[p];
                auto& templates=cell.transfer_templates[p];templates.reserve(source.cohorts_.size());
                for(std::size_t c=0;c<source.cohorts_.size();++c) {
                    const auto found=std::find(source.expert_to_cohort_.begin(),source.expert_to_cohort_.end(),int(c));
                    if(found==source.expert_to_cohort_.end()){templates.emplace_back();continue;}
                    const int e=int(found-source.expert_to_cohort_.begin());const auto& cohort=source.cohorts_[c];
                    TransferTemplate prepared{cell.fused->expert_view(p,e,cohort.bytes_per_expert,-1)};
                    const auto fields=moe_cache_fields(cohort.active);
                    if(fields.size()!=cohort.bytes_per_expert.size())throw std::logic_error("MFE transfer template field count differs");
                    for(std::size_t f=0;f<fields.size();++f)
                        prepared.bases[f]=fields[f].defined()?static_cast<const uint8_t*>(fields[f].data_ptr()):nullptr;
                    templates.push_back(prepared);
                }
            }
        }
        cell.transfer_cache=cell.two_stage && cache_->pipeline_transfer_cache_ && cache_->pipeline_ram_registered_bytes_>0;
        cell.mapped_copy=cell.transfer_cache && cache_->pipeline_mapped_copy_;
        cell.phased_transfer=cell.transfer_cache && !cell.mapped_copy &&
            execution.config.moe_ffn_transfer_phases.value_or(mfq::cuda::runtime_options::phased_transfer());
        cell.mapped_overlap=cell.mapped_copy && mfq::cuda::runtime_options::mapped_copy_overlap();
        if(cell.transfer_cache)for(int p=0;p<3;++p)for(auto& cohort:sources_[p]->cohorts_) {
            auto& arena=stage(p,cohort);
            if(!arena.book)arena.book=std::make_unique<mfq::MoeCacheSlotBook>(arena.slots);
            cell.binding[p].stages.push_back(&arena);
        }
        if(!cell.two_stage)for(int projection=0;projection<3;++projection) {
            auto& source=*sources_[projection];auto& binding=cell.binding[projection];
            for(auto* runtime:{&binding.hot,&binding.transfer}) {
                runtime->n_experts=experts_;runtime->out_per_expert=source.cpu_->out_per_expert;
                runtime->neuron_len=source.cpu_->neuron_len;runtime->partial_experts=true;
            }
            for(auto& cohort:source.cohorts_) {
                auto& temporary=stage(projection,cohort);binding.stages.push_back(&temporary);
                auto hot=cohort.active;hot.expert_local=tb::full({experts_},-1,cell.ids.options());
                auto transfer=fields_pool(cohort.active,temporary.fields,temporary.slots,source.cpu_->out_per_expert);
                transfer.expert_local=tb::full({experts_},-1,cell.ids.options());
                auto retained=cohort.active;retained.expert_local=tb::full({experts_},-1,cell.ids.options());
                binding.hot.pools.push_back(std::move(hot));binding.transfer.pools.push_back(std::move(transfer));
                binding.transfer.pools.push_back(std::move(retained));
                binding.owner.push_back(tb::tensor(cohort.expert_to_local).to(tb::kCUDA));
            }
            const auto* compact_mma=std::getenv("MFQ_MOE_PREFILL_COMPACT_MMA");
            const bool dense_prefill=cell.tokens>8 && compact_mma && std::atoi(compact_mma)!=0;
            initialize_mixed_nvq_dispatch(binding.hot,execution.config,dense_prefill);
            initialize_mixed_nvq_dispatch(binding.transfer,execution.config,dense_prefill);
            if(!cell.graph_replay) {
                if(!cache_->pipeline_prefill_activations_)
                    cache_->pipeline_prefill_activations_=std::make_shared<MixedMoeRuntime>();
                const int rows=projection==2?cell.entries:cell.tokens;
                for(auto* runtime:{&binding.hot,&binding.transfer})
                    for(const auto& g:runtime->activation_geometry()) {
                        const MixedMoeTransformKey transform{g.transform_block,g.transform_seed};
                        const MixedMoeActivationKey key{rows,g.groups,g.gs,cell.input.get_device(),transform};
                        runtime->activation_workspaces.insert_or_assign(key,
                            cache_->pipeline_prefill_activations_->activation_workspace(
                                cell.input,rows,g.groups,g.gs,transform));
                    }
            }
        }
        cell.shared_input=!cell.two_stage && execution.config.split_moe_activation_reuse && cell.tokens<=8 &&
            execution.kl_mmq.mode==KlMmqMode::Default && !(execution.config.moe_prefill_mma &&
                !execution.force_moe_prefill_mma_off && cell.tokens>=execution.config.moe_prefill_mma_min_tokens);
        if(cell.shared_input) {
            std::vector<MoeActivationGeometry> geometries;
            for(int projection=0;projection<2;++projection)for(const auto& geometry:cell.binding[projection].hot.activation_geometry()) {
                if(geometry.transform_block || geometry.groups<=0 || geometry.gs<=0 || geometry.gs>64)
                    throw std::runtime_error("shared FFN input needs canonical unrotated NINT/NVQ groups");
                if(std::find(geometries.begin(),geometries.end(),geometry)==geometries.end())geometries.push_back(geometry);
            }
            std::vector<int64_t> descriptors;
            for(const auto& geometry:geometries) {
                auto workspace=cell.binding[0].hot.activation_workspace(cell.input,cell.tokens,geometry.groups,geometry.gs,{});
                const MixedMoeActivationKey key{cell.tokens,geometry.groups,geometry.gs,cell.input.get_device(),{}};
                for(int projection=0;projection<2;++projection)for(auto* runtime:{&cell.binding[projection].hot,&cell.binding[projection].transfer})
                    runtime->activation_workspaces.insert_or_assign(key,workspace);
                descriptors.insert(descriptors.end(),{reinterpret_cast<int64_t>(workspace.qx.data_ptr()),
                    reinterpret_cast<int64_t>(workspace.xscale.data_ptr()),geometry.groups,geometry.gs});
                cell.shared_input_groups+=geometry.groups;
            }
            if(descriptors.empty())throw std::runtime_error("shared FFN input has no quantization geometry");
            cell.shared_input_descriptors=tb::tensor(descriptors).reshape({int64_t(geometries.size()),4}).to(tb::kCUDA);
        }
        std::vector<int64_t> dispatch;
        const auto pointer=[&](const auto* p){dispatch.push_back(reinterpret_cast<int64_t>(p));};
        if(!cell.two_stage)for(int i=0;i<3;++i)for(std::size_t c=0;c<sources_[i]->cohorts_.size();++c) {
            const auto& binding=cell.binding[i];
            pointer(sources_[i]->cohorts_[c].active.expert_local.data_ptr<int32_t>());
            pointer(binding.owner[c].data_ptr<int32_t>());
            pointer(cell.local.d<int32_t>()+i*experts_);
            pointer(binding.hot.pools[c].expert_local.data_ptr<int32_t>());
            pointer(binding.transfer.pools[2*c].expert_local.data_ptr<int32_t>());
            pointer(binding.transfer.pools[2*c+1].expert_local.data_ptr<int32_t>());
        }
        if(!cell.two_stage)cell.dispatch_pointers=tb::tensor(dispatch).to(tb::kCUDA);
        cell.device_abort=tb::empty({1},cell.input.options().dtype(tb::kInt32));
        cell.device_kind=tb::empty({experts_},cell.input.options().dtype(tb::kInt32));
        if(!cache_->pipeline_host_stage_.defined()) {
            std::unordered_map<int,int64_t> layer_bytes,layer_fields;
            int64_t experts=0;
            for(const auto& source:cache_->sources_)if(source->quant_source_) {
                int64_t fields=0;experts=std::max(experts,static_cast<int64_t>(source->n_experts()));
                for(int e=0;e<source->n_experts();++e) {
                    const auto n=static_cast<int64_t>(source->cohorts_[source->expert_to_cohort_[e]].bytes_per_expert.size());
                    layer_bytes[source->layer_id_]+=source->bytes_for_expert(e)+16*n;fields=std::max(fields,n);
                }
                layer_fields[source->layer_id_]+=fields;
            }
            int64_t bytes=0,fields=0;
            for(const auto& item:layer_bytes)bytes=std::max(bytes,item.second);
            for(const auto& item:layer_fields)fields=std::max(fields,item.second);
            bytes+=(32+24*fields*experts+15)&~int64_t(15);
            cache_->pipeline_host_stage_=tb::empty({bytes},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        }
        if(cell.tokens<=8) {
            if(!cache_->pipeline_wire_route_bytes_) {
                std::unordered_map<int,int64_t> layer_bytes,layer_fields;
                for(const auto& source:cache_->sources_)if(source->quant_source_) {
                    int64_t bytes=0,fields=0;
                    for(int e=0;e<source->n_experts();++e) {
                        const auto n=static_cast<int64_t>(source->cohorts_[source->expert_to_cohort_[e]].bytes_per_expert.size());
                        bytes=std::max(bytes,source->bytes_for_expert(e)+16*n);fields=std::max(fields,n);
                    }
                    layer_bytes[source->layer_id_]+=bytes;layer_fields[source->layer_id_]+=fields;
                }
                for(const auto& item:layer_bytes)cache_->pipeline_wire_route_bytes_=std::max(cache_->pipeline_wire_route_bytes_,item.second);
                for(const auto& item:layer_fields)cache_->pipeline_wire_route_fields_=std::max(cache_->pipeline_wire_route_fields_,item.second);
            }
            const auto routes=std::min(cell.entries,experts_);
            cell.wire_header_bytes=(32+24*routes*cache_->pipeline_wire_route_fields_+15)&~int64_t(15);
            const auto capacity=cell.wire_header_bytes+routes*cache_->pipeline_wire_route_bytes_;
            if(!cache_->pipeline_wire_gpu_.defined() || cache_->pipeline_wire_gpu_.numel()<capacity)
                cache_->pipeline_wire_gpu_=tb::empty({capacity},cell.input.options().dtype(tb::kUInt8));
            cell.wire=cache_->pipeline_wire_gpu_;
            MFQ_CUDA_CHECK(cudaMemsetAsync(cell.wire.data_ptr(),0,32,stream_));
            MFQ_CUDA_CHECK(mfq::cuda::moe_wire_launch_geometry(&cell.wire_blocks,&cell.wire_threads));
        }
        if(cell.mapped_copy) {
            // The first decode cell can follow an ordinary prefill DMA.
            // Later graph-owned copies are fenced by the next route export.
            if(cache_->transfer_ready_recorded_) {
                MFQ_CUDA_CHECK(cudaEventSynchronize(cache_->transfer_ready_));
                cache_->transfer_ready_recorded_=false;cache_->pipeline_dma_leases_.clear();
            }
            if(!cache_->pipeline_ram_alias_)MFQ_CUDA_CHECK(cudaHostGetDevicePointer(
                &cache_->pipeline_ram_alias_,cache_->pipeline_ram_complement_->data_ptr(),0));
            if(!cache_->pipeline_host_stage_alias_)MFQ_CUDA_CHECK(cudaHostGetDevicePointer(
                &cache_->pipeline_host_stage_alias_,cache_->pipeline_host_stage_.data_ptr(),0));
            const auto fields=std::size_t(cell.entries)*cache_->pipeline_wire_route_fields_+1;
            cell.mapped_copy_table=std::make_unique<Mapped>(32+fields*sizeof(mfq::cuda::MoeMappedCopyDescriptor));
            cell.mapped_copy_snapshot=tb::empty({int64_t(cell.mapped_copy_table->bytes)},
                cell.input.options().dtype(tb::kUInt8));
            if(cell.mapped_overlap)cell.copy_events=std::make_unique<CopyEvents>();
            MFQ_CUDA_CHECK(mfq::cuda::moe_mapped_copy_launch_geometry(&cell.wire_blocks,&cell.wire_threads));
        }
        if(!cache_->pipeline_pool_context_) {
            cache_->pipeline_pool_context_=mfq::cuda::default_context(cell.input.get_device());
            cache_->pipeline_pool_stream_=stream_;cache_->pipeline_pool_context_->begin_graph_pool(stream_);
        } else if(cache_->pipeline_pool_stream_!=stream_)throw std::runtime_error("MFQ graph buffers cannot cross execution streams");
        for(int flag=0;flag<4;++flag)publish(cell.flags.h<uint32_t>()+16*flag);
        if(cell.phased_transfer)publish(cell.flags.h<uint32_t>()+80);
        if(cell.phased_transfer && mfq::cuda::runtime_options::dma_graph_batch() &&
                !cache_->pipeline_dma_batches_[0]) {
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(cache_->weight_stream_));
            auto* dummy=cache_->pipeline_host_stage_.data_ptr<uint8_t>();
            *dummy=0;
            for(int phase=0;phase<2;++phase) {
                cache_->pipeline_dma_batches_[phase]=std::make_unique<mfq::cuda::DmaCopyBatch>(cache_->weight_stream_);
                cache_->pipeline_dma_batches_[phase]->prepare(32,cell.wire.data_ptr(),dummy);
            }
        }
        if(cache_->pipeline_dma_profile_enabled_ && !cell.mapped_copy && !cell.phased_transfer) {
            try {
                if(cache_->pipeline_dma_profile_fail_alloc_)throw std::bad_alloc();
                cell.dma_timing=std::make_unique<DmaTimes>();
            } catch(const std::bad_alloc&) {
                cache_->pipeline_dma_profile_enabled_=false;cache_->pipeline_dma_profile_dropped_=true;
            }
        }
        if(window)return;
        if(!cell.graph_replay) {
            // Prime format metadata and activation workspaces with all waits
            // released. Eager prefill temporaries are stream-ordered and must
            // not be retained by the decode graph pool.
            launch_body(cell,execution);
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));cell.output={};
            return;
        }
        // Warm the allocation/workspace shape, then capture exactly the same
        // body. Plans disable every expert during this preparation.
        {mfq::cuda::GraphWarmupScope warmup(cache_->pipeline_pool_context_,stream_);launch_body(cell,execution);}
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));cell.output={};
        std::string error;
        if(!cell.graphs->record(1,cell.tokens,[&]{launch_body(cell,execution);},error))throw std::runtime_error(error);
    }
    tb::Tensor gpu_ffn(Cell& cell,CudaExecutionContext& execution,bool transfer,const MoeRoutePlan& route) {
        auto projection=[&](int i,const tb::Tensor& x) {
            auto& binding=cell.binding[i];auto& runtime=transfer ? binding.transfer : binding.hot;
            return runtime.forward(execution.config,execution.kl_mmq,execution.force_moe_prefill_mma_off,
                execution.force_moe_pool_path,x,route,cell.shared_input && i<2);
        };
        auto gate_output=projection(0,cell.input),up_output=projection(1,cell.input);
        if(cell.fused_activation)
            return projection(2,moe_swiglu_rounded_cuda(gate_output,up_output,cache_->pipeline_sigmoid_table_));
        auto gu=tb::cat({gate_output,up_output},-1);
        auto gate=gu.narrow(-1,0,ff_),up=gu.narrow(-1,ff_,ff_);
        return projection(2,(gate*tb::sigmoid(gate))*up);
    }
    void launch_body(Cell& cell,CudaExecutionContext& execution,const tb::Tensor* input_view=nullptr,
            const tb::Tensor* ids_view=nullptr,const tb::Tensor* weights_view=nullptr) {
        const auto& input=input_view?*input_view:cell.input;
        const auto& ids=ids_view?*ids_view:cell.ids;
        if(cell.two_stage) {
            if(input_view)cell.fused->bind_window_inputs(input,ids,*weights_view);
            else cell.fused->use_staged_inputs();
        }
        if(auto* window=mfq::cuda::DecodeWindow::recording()) {
            const auto found=prediction_cells_.find(window);
            if(found!=prediction_cells_.end()) {
                auto& prediction=*found->second;
                mfq::cuda::wait_mapped_plan(prediction.flags.d<uint32_t>()+4,nullptr,nullptr,0,
                    prediction.flags.d<uint32_t>()+8,reinterpret_cast<uint32_t*>(prediction.aborted.data_ptr<int32_t>()),stream_);
                mfq::cuda::moe_scatter_wire(cache_->pipeline_prefetch_wire_gpu_.data_ptr(),
                    reinterpret_cast<uint32_t*>(prediction.aborted.data_ptr<int32_t>()),cell.wire_blocks,cell.wire_threads,stream_);
            }
        }
        const auto copy_mapped=[&](cudaStream_t copy_stream) {
            auto* snapshot=cell.mapped_copy_snapshot.data_ptr();
            const auto capacity=(cell.mapped_copy_table->bytes-32)/sizeof(mfq::cuda::MoeMappedCopyDescriptor);
            mfq::cuda::moe_snapshot_mapped(cell.mapped_copy_table->device,cell.flags.d<uint32_t>()+64,
                snapshot,capacity,copy_stream);
            mfq::cuda::moe_copy_mapped(snapshot,reinterpret_cast<const uint32_t*>(snapshot)+4,
                cell.wire_blocks,cell.wire_threads,copy_stream);
        };
        const auto mark=[&](const char* phase) {
            if(auto* window=mfq::cuda::DecodeWindow::recording())window->mark(sources_[0]->layer_id_,phase);
        };
        mark("expert_begin");
        cudaStreamCaptureStatus capture;
        MFQ_CUDA_CHECK(cudaStreamIsCapturing(stream_,&capture));
        const auto event_flags=capture==cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault;
        const auto measure=[&](int point) {
            if(!cell.skip_gpu_timing)
                MFQ_CUDA_CHECK(cudaEventRecordWithFlags(cell.timing.points[point],stream_,event_flags));
        };
        mfq::cuda::moe_publish_routes(input.data_ptr(),ids.data_ptr<int32_t>(),cell.x.d<float>(),
            cell.host_ids.d<int32_t>(),cell.flags.d<uint32_t>(),cell.tokens*width_,cell.entries,stream_);
        if(cell.two_stage) {
            if(cell.mapped_overlap) {
                MFQ_CUDA_CHECK(cudaEventRecord(cell.copy_events->fork,stream_));
                MFQ_CUDA_CHECK(cudaStreamWaitEvent(cache_->weight_stream_,cell.copy_events->fork,0));
                mfq::cuda::wait_mapped_flag(cell.flags.d<uint32_t>()+32,cache_->weight_stream_);
                copy_mapped(cache_->weight_stream_);
                MFQ_CUDA_CHECK(cudaEventRecord(cell.copy_events->done,cache_->weight_stream_));
            }
            if(shared_)cell.shared_gate.copy_(shared_weights_.gate(execution,input));
            mark("shared_end");
            cell.fused->prepare();
            // Resident arithmetic runs while cold weights are in flight.
            // Keep DMA waits outside the two arithmetic event intervals.
            measure(0);mark("hot_begin");
            cell.fused->resident();
            measure(1);mark("hot_end");
            if(cell.mapped_overlap)MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream_,cell.copy_events->done,0));
            cell.fused->wait_transfer();
            if(cell.mapped_copy && !cell.mapped_overlap)copy_mapped(stream_);
            if(cell.phased_transfer)mfq::cuda::moe_scatter_wire_phase(cell.wire.data_ptr(),cell.fused->batch().aborted,
                cell.wire_blocks,cell.wire_threads,1,stream_);
            else if(cell.transfer_cache && !cell.mapped_copy)mfq::cuda::moe_scatter_wire(cell.wire.data_ptr(),cell.fused->batch().aborted,
                cell.wire_blocks,cell.wire_threads,stream_);
            measure(2);mark("dma_end");
            cell.fused->transferred();
            mark("transfer_gate_up_end");
            if(cell.phased_transfer) {
                mfq::cuda::wait_mapped_plan(cell.flags.d<uint32_t>()+80,nullptr,nullptr,0,
                    cell.flags.d<uint32_t>()+64,cell.fused->batch().aborted,stream_);
                mfq::cuda::moe_scatter_wire_phase(cell.wire.data_ptr(),cell.fused->batch().aborted,
                    cell.wire_blocks,cell.wire_threads,2,stream_);
            }
            mark("cpu_wait_begin");
            if(!cell.skip_cpu_wait)cell.fused->wait_cpu();
            mark("cpu_wait_end");
            cell.fused->down_reduce_compute();
            measure(3);mark("transfer_end");
            cell.output=cell.fused->output();mark("cpu_end");mark("expert_end");return;
        }
        if(cell.shared_input)moe_quantize_shared_input_cuda(cell.input,cell.shared_input_descriptors,cell.shared_input_groups);
        auto shared=shared_ ? shared_(execution,cell.input) : tb::Tensor{};
        mark("shared_end");
        if(cell.gpu_resident_dispatch) {
            // Residency publication precedes this graph on its compute stream.
            // A complete GPU bundle is never assigned to CPU by the host plan.
            mfq::cuda::moe_classify_resident(reinterpret_cast<const uint64_t*>(cell.dispatch_pointers.data_ptr<int64_t>()),
                static_cast<int>(sources_[0]->cohorts_.size()),static_cast<int>(sources_[1]->cohorts_.size()),
                static_cast<int>(sources_[2]->cohorts_.size()),experts_,cell.device_kind.data_ptr<int32_t>(),
                reinterpret_cast<uint32_t*>(cell.device_abort.data_ptr<int32_t>()),stream_);
        } else {
            mfq::cuda::wait_mapped_plan(cell.flags.d<uint32_t>()+16,cell.kind.d<int32_t>(),cell.device_kind.data_ptr<int32_t>(),
                experts_,cell.flags.d<uint32_t>()+64,reinterpret_cast<uint32_t*>(cell.device_abort.data_ptr<int32_t>()),stream_);
        }
        const auto maps=[&] {
            mfq::cuda::moe_update_dispatch_maps(reinterpret_cast<const uint64_t*>(cell.dispatch_pointers.data_ptr<int64_t>()),
                static_cast<int>(cell.dispatch_pointers.numel()/6),experts_,cell.device_kind.data_ptr<int32_t>(),
                reinterpret_cast<const uint32_t*>(cell.device_abort.data_ptr<int32_t>()),stream_);
        };
        maps();
        const auto route=build_moe_route_plan(cell.ids,experts_);
        MFQ_CUDA_CHECK(cudaEventRecordWithFlags(cell.timing.points[0],stream_,event_flags));
        mark("hot_begin");
        auto hot=gpu_ffn(cell,execution,false,route);
        MFQ_CUDA_CHECK(cudaEventRecordWithFlags(cell.timing.points[1],stream_,event_flags));
        mark("hot_end");
        // DMA completion also publishes the host CPU/transfer choices. Partial
        // GPU bundles remain on that path, with their resident projections retained.
        mfq::cuda::wait_mapped_plan(cell.flags.d<uint32_t>()+32,
            cell.gpu_resident_dispatch?cell.kind.d<int32_t>():nullptr,
            cell.gpu_resident_dispatch?cell.device_kind.data_ptr<int32_t>():nullptr,
            cell.gpu_resident_dispatch?experts_:0,cell.flags.d<uint32_t>()+64,
            reinterpret_cast<uint32_t*>(cell.device_abort.data_ptr<int32_t>()),stream_);
        // Re-read cancellation after DMA completion, before kernels may use
        // transferred fields. A failed host phase can safely release the graph.
        MFQ_CUDA_CHECK(cudaEventRecordWithFlags(cell.timing.points[2],stream_,event_flags));
        mark("dma_end");
        maps();
        if(cell.wire.defined())mfq::cuda::moe_scatter_wire(cell.wire.data_ptr(),reinterpret_cast<const uint32_t*>(cell.device_abort.data_ptr<int32_t>()),
            cell.wire_blocks,cell.wire_threads,stream_);
        auto transfer=gpu_ffn(cell,execution,true,route);
        MFQ_CUDA_CHECK(cudaEventRecordWithFlags(cell.timing.points[3],stream_,event_flags));
        mark("transfer_end");
        mfq::cuda::wait_mapped_flag(cell.flags.d<uint32_t>()+48,stream_);
        mark("cpu_end");
        auto parts=tb::empty({cell.tokens,cell.routes,width_},cell.input.options());
        mfq::cuda::moe_merge_results(hot.data_ptr(),transfer.data_ptr(),cell.cpu.device,cell.ids.data_ptr<int32_t>(),
            cell.device_kind.data_ptr<int32_t>(),parts.data_ptr(),cell.entries,width_,stream_);
        cell.output=shared.defined() ? moe_weighted_reduce_cuda(parts,cell.weights)+shared : parts;
        mark("expert_end");
        MFQ_CUDA_CHECK(cudaGetLastError());
    }
    void wait_publication(Cell& cell) {
        const auto* flag=cell.flags.h<uint32_t>();
        auto flush=std::chrono::steady_clock::now();const auto start=flush;uint32_t spins=0;
        const auto poll_interval=std::chrono::microseconds(mfq::cuda::runtime_options::route_poll_interval_us());
        if(*(volatile const uint32_t*)flag<1)mfq::cuda::check_route_submission(stream_);
        while(*(volatile const uint32_t*)flag<1) {
            _mm_pause();if((++spins&1023u)!=0)continue;
            const auto now=std::chrono::steady_clock::now();
            if(now-flush>=poll_interval) {
                flush=now;const auto q=cudaStreamQuery(stream_);
                if(q!=cudaSuccess && q!=cudaErrorNotReady)MFQ_CUDA_CHECK(q);
            }
            if(now-start>std::chrono::seconds(60)) {
                trace_serve(cell,"route_timeout",true);
                throw std::runtime_error("MFQ GPU route publication stalled");
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
    }
    void plan(Cell& cell) {
        const bool must_drain=!cell.in_window || !cache_->pipeline_dispatch_replay_.empty();
        const bool calibration_busy=cache_->pipeline_cpu_calibration_.collect(cache_->pipeline_cpu_cost_,must_drain);
        const bool shadow=cell.in_window && cache_->pipeline_shared_cpu_cost_ &&
            cache_->pipeline_cpu_transfer_budget_ && mfq::cuda::runtime_options::background_calibration() && cache_->pipeline_dispatch_replay_.empty();
        cell.calibration_expert=-1;
        std::vector<int32_t> legacy_ids;
        auto& ids=cell.reuse_plan?cell.plan_ids:legacy_ids;
        ids.assign(cell.host_ids.h<int32_t>(),cell.host_ids.h<int32_t>()+cell.entries);
        auto resident=[&](int e) {return std::all_of(sources_.begin(),sources_.end(),[&](const auto& s){return s->quant_source_->gpu_resident(e);});};
        auto eligible=[&](int e) {return std::all_of(sources_.begin(),sources_.end(),[&](const auto& s){return s->quant_source_->host_cache()->contains(s->quant_source_->host_key(e));});};
        const auto transfer_bytes=[&](int e) {
            uint64_t bytes=0;
            for(int p=0;p<3;++p) {
                const auto& source=*sources_[p];const int c=source.expert_to_cohort_[e];
                if(source.quant_source_->gpu_resident(e))continue;
                if(cell.transfer_cache) {
                    const auto* book=cell.binding[p].stages[c]->book.get();
                    if(book && book->slot_for({source.id_,c,e})>=0)continue;
                }
                bytes+=source.bytes_for_expert(e);
            }
            return bytes;
        };
        const auto measured_cpu=[&](const mfq::MoeDispatchGroup& group) {
            return cache_->pipeline_shared_cpu_cost_
                ? cache_->pipeline_cpu_cost_.estimate(cpu_cost_key(group))
                : cpu_work_ ? cpu_ns_/cpu_work_*group.positions.size() : 0;
        };
        const mfq::MoeDispatchRates rates{cache_->pipeline_pcie_gbps_,
            stage_bytes_ ? stage_ns_/stage_bytes_ : 0,gpu_samples_ ? gpu_ns_/gpu_samples_ : 0,
            cache_->pipeline_cpu_transfer_budget_};
        if(cell.reuse_plan)
            mfq::plan_moe_dispatch_reuse(cell.plan,cell.plan_workspace,ids,experts_,0,resident,[](int){return true;});
        else cell.plan=mfq::plan_moe_dispatch(ids,experts_,0,resident,[](int){return true;});
        std::size_t misses=0;for(const auto& group:cell.plan.groups)misses+=group.kind!=mfq::MoeDispatchKind::GpuResident;
        if(cell.tokens>8 || cache_->config_.moe_ram_pcie_fraction) {
            const auto quota=cell.tokens>8 ? misses : static_cast<std::size_t>(misses*cache_->ram_pcie_fraction_);
            if(cell.reuse_plan)mfq::assign_moe_transfer_quota(cell.plan,quota,[](int){return true;});
            else cell.plan=mfq::plan_moe_dispatch(ids,experts_,quota,resident,[](int){return true;});
        } else {
            std::vector<std::size_t> cold;
            std::vector<mfq::MoeDispatchCost> costs;
            for(std::size_t i=0;i<cell.plan.groups.size();++i) {
                auto& group=cell.plan.groups[i];if(group.kind==mfq::MoeDispatchKind::GpuResident)continue;
                const auto bytes=transfer_bytes(group.expert);
                cold.push_back(i);
                const double measured=measured_cpu(group);
                costs.push_back({measured,bytes,bytes>0 && eligible(group.expert)});
                group.kind=mfq::MoeDispatchKind::GpuTransfer;
            }
            std::vector<bool> selected(costs.size(),false);
            if(cache_->pipeline_shared_cpu_cost_) {
                // Acquire three observations per compatible FFN geometry so
                // one cold worker wake cannot establish the session's rate.
                std::size_t sample=costs.size();
                for(std::size_t i=0;i<costs.size();++i)if(costs[i].cpu_eligible && !costs[i].cpu_ns &&
                    (sample==costs.size() || costs[i].transfer_bytes>costs[sample].transfer_bytes))sample=i;
                if(sample<costs.size()) {
                    if(shadow) {
                        if(!calibration_busy)cell.calibration_expert=cell.plan.groups[cold[sample]].expert;
                    }else selected[sample]=true;
                }
                else {
                    for(auto& cost:costs)cost.cpu_eligible=cost.cpu_eligible && cost.cpu_ns>0;
                    if(!calibration_busy)selected=mfq::select_moe_cpu_work(costs,rates);
                }
            } else if(cpu_work_) {
                selected=mfq::select_moe_cpu_work(costs,rates);
            } else {
                // One retained production group supplies the first CPU rate;
                // every other cold group uses the calibrated PCIe path.
                std::size_t sample=costs.size();
                for(std::size_t i=0;i<costs.size();++i)if(costs[i].cpu_eligible &&
                    (sample==costs.size() || costs[i].transfer_bytes>costs[sample].transfer_bytes))sample=i;
                if(sample<costs.size())selected[sample]=true;
            }
            for(std::size_t i=0;i<cold.size();++i)if(selected[i])cell.plan.groups[cold[i]].kind=mfq::MoeDispatchKind::Cpu;
        }
        if(!cache_->pipeline_dispatch_replay_.empty()) {
            if(cache_->pipeline_dispatch_replay_cursor_>=cache_->pipeline_dispatch_replay_.size())
                throw std::runtime_error("MFQ dispatch replay ended before inference");
            const auto& replay=cache_->pipeline_dispatch_replay_[cache_->pipeline_dispatch_replay_cursor_++];
            if(replay.layer!=sources_[0]->layer_id_ || replay.tokens!=cell.tokens || replay.ids!=ids) {
                std::ostringstream message;
                message<<"MFQ dispatch replay input/route mismatch record="
                    <<cache_->pipeline_dispatch_replay_cursor_-1
                    <<" layer="<<sources_[0]->layer_id_<<" expected_layer="<<replay.layer
                    <<" tokens="<<cell.tokens<<" expected_tokens="<<replay.tokens
                    <<" routes="<<ids.size()<<" expected_routes="<<replay.ids.size();
                const auto count=std::min(ids.size(),replay.ids.size());
                for(std::size_t position=0;position<count;++position)if(ids[position]!=replay.ids[position]) {
                    message<<" position="<<position<<" expert="<<ids[position]
                        <<" expected_expert="<<replay.ids[position];
                    break;
                }
                throw std::runtime_error(message.str());
            }
            std::size_t matched=0;
            for(auto& group:cell.plan.groups) {
                const bool cpu=std::find(replay.cpu.begin(),replay.cpu.end(),group.expert)!=replay.cpu.end();
                if(cpu && (resident(group.expert) || !eligible(group.expert)))
                    throw std::runtime_error("MFQ dispatch replay CPU expert changed tier ownership");
                matched+=cpu;
                group.kind=cpu?mfq::MoeDispatchKind::Cpu:resident(group.expert)?mfq::MoeDispatchKind::GpuResident:mfq::MoeDispatchKind::GpuTransfer;
            }
            if(matched!=replay.cpu.size())throw std::runtime_error("MFQ dispatch replay CPU expert missing from routes");
        }
        // Explicit quotas also share the CPU pool. An unfinished calibration
        // cannot be followed by inference work queued behind that calibration.
        if(calibration_busy)for(auto& group:cell.plan.groups)
            if(group.kind==mfq::MoeDispatchKind::Cpu)group.kind=mfq::MoeDispatchKind::GpuTransfer;
        std::memset(cell.kind.host,0,cell.kind.bytes);
        // Only the unfused dispatch map consumes local cohort ordinals.
        if(!cell.reuse_plan)std::memset(cell.local.host,0xff,cell.local.bytes);
        const bool early_gu=cell.two_stage && cell.fused->batch().resident_plan_overlap &&
            mfq::cuda::runtime_options::early_gate_up() && mfq::cuda::runtime_options::parallel_gate_up();
        cache_->stats_.pipeline_early_gate_up_enabled=early_gu;
        for(auto& group:cell.plan.groups) {
            // Mixed formats have projection-granular residency. A partial
            // GPU bundle must use its remaining RAM projections on the GPU.
            if(group.kind==mfq::MoeDispatchKind::Cpu && !eligible(group.expert))group.kind=mfq::MoeDispatchKind::GpuTransfer;
            const int kind=group.kind==mfq::MoeDispatchKind::Cpu ? 0 : group.kind==mfq::MoeDispatchKind::GpuResident ? 1 : 2;
            cell.kind.h<int32_t>()[group.expert]=kind;
            if(cell.two_stage && kind==2 && sources_[0]->quant_source_->gpu_resident(group.expert) &&
                sources_[1]->quant_source_->gpu_resident(group.expert) && !sources_[2]->quant_source_->gpu_resident(group.expert)) {
                ++cache_->stats_.pipeline_gate_up_primary_down_missing_experts;
                cache_->stats_.pipeline_gate_up_primary_down_missing_positions+=group.positions.size();
                if(early_gu)cache_->stats_.pipeline_early_gate_up_positions+=group.positions.size();
            }
            for(auto pos:group.positions)cell.plan.kinds[pos]=group.kind;
        }
        if(!cell.reuse_plan)for(int projection=0;projection<3;++projection) {
            const auto& source=*sources_[projection];std::vector<int> slots(source.cohorts_.size());
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                const auto e=group.expert,c=source.expert_to_cohort_[e];cell.local.h<int32_t>()[projection*experts_+e]=slots[c]++;
            }
        }
        if(cell.two_stage) {
            std::memset(cell.transfer_index->host,0xff,cell.transfer_index->bytes);cell.transfer_views.clear();
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                cell.transfer_index->h<int32_t>()[group.expert]=int32_t(cell.transfer_views.size()/3);
                for(int p=0;p<3;++p) {
                    const auto& source=*sources_[p];const int c=source.expert_to_cohort_[group.expert];
                    const auto& cohort=source.cohorts_[c];const int slot=cohort.arena->book->slot_for({source.id_,c,group.expert});
                    if(cell.reuse_plan) {
                        const auto& prepared=cell.transfer_templates[p][c];auto view=prepared.view;
                        if(slot>=0)for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f)
                            view.fields[f]=prepared.bases[f]?prepared.bases[f]+int64_t(slot)*cohort.bytes_per_expert[f]:nullptr;
                        if(cell.check_reused_views) {
                            const auto original=cell.fused->expert_view(p,group.expert,cohort.bytes_per_expert,slot);
                            if(std::memcmp(&original,&view,sizeof(view)))throw std::logic_error("MFE reused transfer metadata differs from original");
                        }
                        cell.transfer_views.push_back(view);
                    } else cell.transfer_views.push_back(cell.fused->expert_view(p,group.expert,cohort.bytes_per_expert,slot));
                }
            }
        }
        if(cache_->pipeline_dispatch_capture_required_ || !cache_->pipeline_dispatch_record_path_.empty()) {
            try {
                MoeExpertCache::DispatchRecord row;row.layer=sources_[0]->layer_id_;row.tokens=cell.tokens;row.ids=ids;
                for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::Cpu)row.cpu.push_back(group.expert);
                if(cell.tokens==1 && !cache_->pipeline_miss_record_path_.empty()) {
                    row.missing_bytes.resize(experts_,0);
                    for(int e=0;e<experts_;++e)for(int p=0;p<3;++p) {
                        const auto& source=*sources_[p];const int c=source.expert_to_cohort_[e];
                        if(source.quant_source_->gpu_resident(e))continue;
                        const auto* book=cell.transfer_cache?cell.binding[p].stages[c]->book.get():nullptr;
                        if(book && book->slot_for({source.id_,c,e})>=0)continue;
                        row.missing_bytes[e]+=source.bytes_for_expert(e);
                    }
                }
                cache_->pipeline_dispatch_records_.push_back(std::move(row));
            }catch(const std::bad_alloc&) {
                if(cache_->pipeline_dispatch_capture_required_)throw;
                cache_->pipeline_dispatch_record_path_.clear();cache_->pipeline_dispatch_records_.clear();
            }
        }
        if(cell.skip_cpu_wait) {
            if(std::any_of(cell.plan.groups.begin(),cell.plan.groups.end(),[](const auto& group) {
                return group.kind==mfq::MoeDispatchKind::Cpu;
            }))throw std::logic_error("GPU-only CPU-wait graph received CPU expert work");
            ++cache_->stats_.pipeline_gpu_only_cpu_wait_skips;
        }
        if(cell.skip_gpu_timing) {
            if(std::any_of(cell.plan.groups.begin(),cell.plan.groups.end(),[](const auto& group) {
                return group.kind==mfq::MoeDispatchKind::Cpu;
            }))throw std::logic_error("fixed GPU timing graph received CPU expert work");
            ++cache_->stats_.pipeline_fixed_gpu_timing_skips;
        }
        if(observer_)observer_(cell.plan);
        publish(cell.flags.h<uint32_t>()+16);
        // GPU-only routes consume no CPU pair buffer. Their CPU completion
        // is independent of DMA submission and can be published with the plan.
        // Every graph replay resets this flag before receiving its new routes.
        if(mfq::cuda::runtime_options::early_no_cpu_ready() &&
            std::none_of(cell.plan.groups.begin(),cell.plan.groups.end(),[](const auto& group) {
                return group.kind==mfq::MoeDispatchKind::Cpu;
            })) {
            publish(cell.flags.h<uint32_t>()+48);
            ++cache_->stats_.pipeline_early_no_cpu_ready_serves;
        }
    }
    void copy_transfer(Cell& cell,void* destination,const void* source,std::size_t bytes,
            std::size_t source_pitch=0,std::size_t rows=1) {
        const auto total_bytes=bytes*rows;
        const bool trace=cache_->pipeline_dma_profile_enabled_ && cell.dma_timing;
        const bool first=trace && !cell.dma_timing->started;
        if(trace) {
            auto& timing=*cell.dma_timing;
            if(first) {
                timing.enqueue_started=std::chrono::steady_clock::now();
                timing.sample.prepare_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    timing.enqueue_started-timing.fetch_started).count();
                timing.sample.first_copy_bytes=total_bytes;
            }
            ++timing.sample.copies;timing.sample.copy_bytes+=total_bytes;
        }
        if(rows==1)MFQ_CUDA_CHECK(cudaMemcpyAsync(destination,source,bytes,cudaMemcpyHostToDevice,cache_->weight_stream_));
        else MFQ_CUDA_CHECK(cudaMemcpy2DAsync(destination,bytes,source,source_pitch,bytes,rows,
            cudaMemcpyHostToDevice,cache_->weight_stream_));
        if(first) {
            // A timing event ahead of the first H2D stalls the concurrent
            // graph/producer protocol on the verified WDDM native workload.
            // Start after the original first copy; its bytes are reported
            // separately and its duration is excluded from the copy interval.
            MFQ_CUDA_CHECK(cudaEventRecord(cell.dma_timing->points[0],cache_->weight_stream_));
            cell.dma_timing->started=true;
        }
    }
    void queue_transfer_ready(Cell& cell,int offset) {
        if(cell.ready_writer) {
            const auto result=cell.ready_writer(reinterpret_cast<CUstream>(cache_->weight_stream_),
                reinterpret_cast<CUdeviceptr>(cell.flags.d<uint32_t>()+offset),1u,0u);
            if(result==CUDA_SUCCESS) {++cache_->stats_.pipeline_transfer_stream_writes;return;}
            if(result==CUDA_ERROR_NOT_SUPPORTED)cell.ready_writer=nullptr;
            else throw std::runtime_error("CUDA transfer notification failed: "+std::to_string(result));
        }
        MFQ_CUDA_CHECK(cudaLaunchHostFunc(cache_->weight_stream_,
            [](void* p){publish(static_cast<uint32_t*>(p));},cell.flags.h<uint32_t>()+offset));
        ++cache_->stats_.pipeline_transfer_host_callbacks;
    }
    void finish_transfer(Cell& cell,bool pending) {
        auto* flag=cell.flags.h<uint32_t>()+(cell.phased_transfer?80:32);
        if(cell.mapped_copy) {
            // The graph owns completion and joins its copy stream before use.
            // A following route publication proves these RAM leases are idle.
            cache_->transfer_ready_recorded_=false;publish(flag);return;
        }
        const bool trace=cache_->pipeline_dma_profile_enabled_ && cell.dma_timing && cell.dma_timing->started;
        if(trace) {
            auto& timing=*cell.dma_timing;
            timing.sample.enqueue_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-timing.enqueue_started).count();
            MFQ_CUDA_CHECK(cudaEventRecord(timing.points[1],cache_->weight_stream_));
        }
        if(!pending)publish(flag);
        else queue_transfer_ready(cell,cell.phased_transfer?80:32);
        if(trace)MFQ_CUDA_CHECK(cudaEventRecord(cell.dma_timing->points[2],cache_->weight_stream_));
        MFQ_CUDA_CHECK(cudaEventRecord(cache_->transfer_ready_,cache_->weight_stream_));
        cache_->transfer_ready_recorded_=true;
        if(pending)mfq::cuda::check_stream_progress(cache_->weight_stream_);
    }
    int64_t upload_transfer_views(Cell& cell,std::vector<mfq::cuda::DmaCopy>* batch=nullptr) {
        if(!cell.two_stage || cell.transfer_views.empty())return 0;
        const int64_t bytes=int64_t(cell.transfer_views.size())*sizeof(mfq::cuda::MfePackedProjection);
        if(bytes>int64_t(cell.transfer_descriptors->bytes) || bytes>cell.fused->transfer_buffer().numel())
            throw std::logic_error("MFE transfer descriptor capacity exceeded");
        for(const auto& view:cell.transfer_views)for(int f=0;f<(view.family==1?7:5);++f)
            if(!view.fields[f] && !(view.family==2 && ((f<3 && !view.sizes[f]) || f==4)))
                throw std::logic_error("MFE transfer descriptor field not published");
        std::memcpy(cell.transfer_descriptors->host,cell.transfer_views.data(),std::size_t(bytes));
        if(!cell.mapped_copy) {
            if(batch)batch->push_back({cell.fused->transfer_buffer().data_ptr(),cell.transfer_descriptors->host,std::size_t(bytes)});
            else copy_transfer(cell,cell.fused->transfer_buffer().data_ptr(),cell.transfer_descriptors->host,std::size_t(bytes));
        }
        return bytes;
    }
    void fetch_registered(Cell& cell) {
        struct Field {uint8_t* destination;tb::Tensor source;int64_t bytes,wire_offset=0;int projection=0,expert=0,field=0;};
        struct Interval {const uint8_t* source;int64_t bytes,offset;};
        std::vector<Field> fields;
        struct CacheRollback {
            std::vector<MoeCacheNewLease> added;
            bool published=false;
            ~CacheRollback(){if(!published)for(const auto& item:added)item.book->discard(item.key,item.slot,item.generation);}
        } rollback;
        // Protect the whole routed working set before any miss can evict a
        // cached field belonging to a later expert in this same invocation.
        if(cell.transfer_cache)for(int p=0;p<3;++p) {
            const auto& source=*sources_[p];
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                const int e=group.expert,c=source.expert_to_cohort_[e];
                if(source.cohorts_[c].arena->book->slot_for({source.id_,c,e})>=0)continue;
                auto* book=cell.binding[p].stages[c]->book.get();const mfq::MoeCacheKey key{source.id_,c,e};
                const int slot=book->slot_for(key);
                if(slot>=0) {
                    cache_->pipeline_transfer_slots_.push_back({book,key,slot,0});
                    book->mark_inflight(slot);
                }
            }
        }
        for(int projection=0;projection<3;++projection) {
            auto& source=*sources_[projection];std::vector<int> slots(source.cohorts_.size());
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                const int e=group.expert,c=source.expert_to_cohort_[e];int slot=slots[c]++;
                const auto& cohort=source.cohorts_[c];
                auto* stage=!cell.two_stage || cell.transfer_cache?cell.binding[projection].stages[c]:nullptr;
                if(stage && slot>=stage->slots)throw std::logic_error("MFQ direct transfer exceeds cohort geometry");
                if(cohort.arena->book->slot_for({source.id_,c,e})>=0)continue;
                if(cell.transfer_cache) {
                    const mfq::MoeCacheKey key{source.id_,c,e};const auto held=stage->book->acquire(key);
                    slot=held.slot;
                    const MoeCacheNewLease item{stage->book.get(),key,slot,held.generation};
                    if(!held.hit)rollback.added.push_back(item);
                    cache_->pipeline_transfer_slots_.push_back(item);stage->book->mark_inflight(slot);
                    const int index=cell.transfer_index->h<int32_t>()[e];
                    for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f)
                        cell.transfer_views[index*3+projection].fields[f]=static_cast<uint8_t*>(stage->fields[f].data_ptr())+
                            int64_t(slot)*cohort.bytes_per_expert[f];
                    if(held.hit) {
                        if(std::find(cache_->pipeline_prefetched_keys_.begin(),cache_->pipeline_prefetched_keys_.end(),key)!=cache_->pipeline_prefetched_keys_.end())
                            cache_->stats_.pipeline_prefetch_hit_bytes+=source.bytes_for_expert(e);
                        ++cache_->stats_.pipeline_transfer_cache_hits;
                        cache_->stats_.pipeline_transfer_cache_saved_bytes+=source.bytes_for_expert(e);
                        continue;
                    }
                    ++cache_->stats_.pipeline_transfer_cache_misses;
                }
                auto lease=source.quant_source_->acquire_expert(e);
                auto values=moe_cache_fields(lease->weights);
                if(values.size()!=cohort.bytes_per_expert.size())throw std::logic_error("MFQ RAM field count changed");
                for(std::size_t f=0;f<values.size();++f) {
                    const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
                    if(bytes!=tensor_nbytes(values[f]))throw std::logic_error("MFQ RAM field geometry changed");
                    fields.push_back({stage?static_cast<uint8_t*>(stage->fields[f].data_ptr())+int64_t(slot)*bytes:nullptr,
                        std::move(values[f]),bytes,0,projection,e,int(f)});
                }
                cache_->pipeline_dma_leases_.push_back(std::move(lease));
            }
        }
        auto* packed=cache_->pipeline_host_stage_.data_ptr<uint8_t>();
        const auto begin=reinterpret_cast<uintptr_t>(cache_->pipeline_ram_complement_->data_ptr());
        const auto size=static_cast<uint64_t>(cache_->pipeline_ram_registered_bytes_);
        if(cell.mapped_copy) {
            auto* header=cell.mapped_copy_table->h<uint64_t>();
            auto* descriptors=reinterpret_cast<mfq::cuda::MoeMappedCopyDescriptor*>(header+4);
            const auto capacity=(cell.mapped_copy_table->bytes-32)/sizeof(*descriptors);
            std::size_t count=0;uint64_t total=0;
            const auto append=[&](void* destination,const void* source,uint64_t bytes) {
                if(!bytes)return;
                if(count==capacity)throw std::logic_error("mapped RAM copy descriptor geometry exceeded");
                total=(total+15)&~uint64_t(15);
                if(bytes>UINT64_MAX-total)throw std::overflow_error("mapped RAM copy byte geometry exceeded");
                descriptors[count++]={reinterpret_cast<uint64_t>(destination),reinterpret_cast<uint64_t>(source),total,bytes};
                total+=bytes;
            };
            int64_t direct_bytes=0,staged_bytes=0;
            for(const auto& field:fields) {
                const auto address=reinterpret_cast<uintptr_t>(field.source.data_ptr());
                const uint8_t* alias=nullptr;
                if(address>=begin && address-begin<=size && uint64_t(field.bytes)<=size-(address-begin)) {
                    alias=static_cast<const uint8_t*>(cache_->pipeline_ram_alias_)+(address-begin);
                    direct_bytes+=field.bytes;
                } else {
                    staged_bytes=(staged_bytes+15)&~int64_t(15);
                    if(field.bytes>cache_->pipeline_host_stage_.numel()-staged_bytes)
                        throw std::logic_error("mapped RAM fallback exceeds pinned stage geometry");
                    std::memcpy(packed+staged_bytes,field.source.data_ptr(),std::size_t(field.bytes));
                    alias=static_cast<const uint8_t*>(cache_->pipeline_host_stage_alias_)+staged_bytes;
                    staged_bytes+=field.bytes;
                }
                append(field.destination,alias,field.bytes);
            }
            const auto descriptor_bytes=upload_transfer_views(cell);
            if(descriptor_bytes)append(cell.fused->transfer_buffer().data_ptr(),cell.transfer_descriptors->device,descriptor_bytes);
            header[0]=count;header[1]=total;header[2]=header[3]=0;
            int64_t payload=0;for(const auto& field:fields)payload+=field.bytes;
            cache_->stats_.ram_pcie_bytes+=payload;
            cache_->stats_.pipeline_direct_ram_bytes+=direct_bytes;
            cache_->stats_.pipeline_staged_ram_bytes+=staged_bytes;
            ++cache_->stats_.pipeline_mapped_copy_serves;
            cache_->stats_.pipeline_mapped_overlap_serves+=cell.mapped_overlap;
            cache_->stats_.pipeline_mapped_copy_bytes+=payload;
            cache_->stats_.pipeline_mapped_copy_descriptors+=count;
            if(cell.in_window)cache_->stats_.pipeline_window_dma_bytes+=payload+descriptor_bytes+32+count*sizeof(*descriptors);
            finish_transfer(cell,true);rollback.published=true;return;
        }

        if(cell.phased_transfer) {
            std::array<std::vector<Interval>,2> intervals;
            std::array<int64_t,2> staged_begin{},staged_end{};
            int64_t offset=cell.wire_header_bytes,direct_bytes=0,staged_bytes=0,gate_up_end=0;
            // A packet boundary separates Gate/Up from Down. Cache ownership,
            // original RAM leases and the final completion event are shared.
            for(int phase=0;phase<2;++phase) {
                std::vector<std::size_t> direct,staged;
                for(std::size_t i=0;i<fields.size();++i) {
                    if((fields[i].projection==2)!=(phase==1))continue;
                    const auto address=reinterpret_cast<uintptr_t>(fields[i].source.data_ptr());
                    if(address>=begin && address-begin<=size && uint64_t(fields[i].bytes)<=size-(address-begin))direct.push_back(i);
                    else staged.push_back(i);
                }
                std::sort(direct.begin(),direct.end(),[&](auto a,auto b) {
                    return reinterpret_cast<uintptr_t>(fields[a].source.data_ptr())<reinterpret_cast<uintptr_t>(fields[b].source.data_ptr());
                });
                for(std::size_t first=0;first<direct.size();) {
                    const auto address=reinterpret_cast<uintptr_t>(fields[direct[first]].source.data_ptr());
                    auto end=address+fields[direct[first]].bytes;auto last=first+1;
                    while(last<direct.size()) {
                        const auto next=reinterpret_cast<uintptr_t>(fields[direct[last]].source.data_ptr());
                        // The wire scatter visits 16-byte boundaries; every
                        // field descriptor must begin on one of them.
                        if(next<end || next-end>15 || ((next-address)&15))break;
                        end=next+fields[direct[last]].bytes;++last;
                    }
                    offset=(offset+15)&~int64_t(15);
                    const auto bytes=static_cast<int64_t>(end-address);
                    if(bytes>cell.wire.numel()-offset)throw std::logic_error("MFQ phased RAM interval exceeds wire geometry");
                    for(auto i=first;i<last;++i)fields[direct[i]].wire_offset=offset+
                        static_cast<int64_t>(reinterpret_cast<uintptr_t>(fields[direct[i]].source.data_ptr())-address);
                    intervals[phase].push_back({reinterpret_cast<const uint8_t*>(address),bytes,offset});
                    offset+=bytes;direct_bytes+=bytes;first=last;
                }
                offset=(offset+15)&~int64_t(15);staged_begin[phase]=offset;
                for(const auto i:staged) {
                    offset=(offset+15)&~int64_t(15);auto& field=fields[i];
                    if(field.bytes>cell.wire.numel()-offset || field.bytes>cache_->pipeline_host_stage_.numel()-offset)
                        throw std::logic_error("MFQ phased exchanged RAM field exceeds wire geometry");
                    std::memcpy(packed+offset,field.source.data_ptr(),std::size_t(field.bytes));
                    field.wire_offset=offset;offset+=field.bytes;staged_bytes+=field.bytes;
                }
                staged_end[phase]=offset;
                if(!phase)gate_up_end=(offset+15)&~int64_t(15);
            }
            if(32+fields.size()*24>std::size_t(cell.wire_header_bytes))
                throw std::logic_error("MFQ phased RAM descriptors exceed route geometry");
            std::sort(fields.begin(),fields.end(),[](const auto& a,const auto& b){return a.wire_offset<b.wire_offset;});
            auto* words=reinterpret_cast<uint64_t*>(packed);
            words[0]=fields.size();words[1]=cell.wire_header_bytes;words[2]=offset;words[3]=gate_up_end;
            int64_t payload_bytes=0;
            for(std::size_t i=0;i<fields.size();++i) {
                words[4+i*3]=reinterpret_cast<uint64_t>(fields[i].destination);
                words[5+i*3]=fields[i].wire_offset;words[6+i*3]=fields[i].bytes;payload_bytes+=fields[i].bytes;
                (fields[i].projection==2?cache_->stats_.pipeline_down_dma_bytes:cache_->stats_.pipeline_gate_up_dma_bytes)+=fields[i].bytes;
            }
            auto* wire=cell.wire.data_ptr<uint8_t>();
            const bool batched=mfq::cuda::runtime_options::dma_graph_batch();
            std::array<std::vector<mfq::cuda::DmaCopy>,2> batches;
            if(batched)batches[0].push_back({wire,packed,std::size_t(cell.wire_header_bytes)});
            else MFQ_CUDA_CHECK(cudaMemcpyAsync(wire,packed,std::size_t(cell.wire_header_bytes),cudaMemcpyHostToDevice,cache_->weight_stream_));
            // Down's immutable format metadata must arrive before hidden-group
            // quantization; its weight fields can remain in flight until use.
            const auto descriptor_bytes=upload_transfer_views(cell,batched?&batches[0]:nullptr);
            int64_t copies=1+(descriptor_bytes>0),wire_bytes=cell.wire_header_bytes+descriptor_bytes;
            for(int phase=0;phase<2;++phase) {
                for(const auto& interval:intervals[phase]) {
                    if(batched)batches[phase].push_back({wire+interval.offset,interval.source,std::size_t(interval.bytes)});
                    else MFQ_CUDA_CHECK(cudaMemcpyAsync(wire+interval.offset,interval.source,
                        std::size_t(interval.bytes),cudaMemcpyHostToDevice,cache_->weight_stream_));
                    ++copies;wire_bytes+=interval.bytes;
                }
                if(staged_end[phase]>staged_begin[phase]) {
                    const auto bytes=staged_end[phase]-staged_begin[phase];
                    if(batched)batches[phase].push_back({wire+staged_begin[phase],packed+staged_begin[phase],std::size_t(bytes)});
                    else MFQ_CUDA_CHECK(cudaMemcpyAsync(wire+staged_begin[phase],packed+staged_begin[phase],
                        std::size_t(bytes),cudaMemcpyHostToDevice,cache_->weight_stream_));
                    ++copies;wire_bytes+=bytes;
                }
                if(batched && !batches[phase].empty()) {
                    auto& batch=cache_->pipeline_dma_batches_[phase];
                    if(batch && batch->submit(batches[phase]))++cache_->stats_.pipeline_dma_graph_launches;
                    else {
                        if(!batch)for(const auto& copy:batches[phase])
                            MFQ_CUDA_CHECK(cudaMemcpyAsync(copy.destination,copy.source,copy.bytes,
                                cudaMemcpyHostToDevice,cache_->weight_stream_));
                        ++cache_->stats_.pipeline_dma_graph_fallbacks;
                    }
                }
                if(!phase)queue_transfer_ready(cell,32);
            }
            cache_->stats_.ram_pcie_bytes+=payload_bytes;
            cache_->stats_.pipeline_dma_copies+=copies;
            cache_->stats_.pipeline_direct_ram_bytes+=direct_bytes;
            cache_->stats_.pipeline_direct_ram_copies+=intervals[0].size()+intervals[1].size();
            cache_->stats_.pipeline_staged_ram_bytes+=staged_bytes;
            if(cell.in_window) {
                cache_->stats_.pipeline_window_dma_copies+=copies;cache_->stats_.pipeline_window_dma_bytes+=wire_bytes;
            }
            finish_transfer(cell,true);rollback.published=true;return;
        }

        std::vector<std::size_t> direct,staged;
        for(std::size_t i=0;i<fields.size();++i) {
            const auto address=reinterpret_cast<uintptr_t>(fields[i].source.data_ptr());
            if(address>=begin && address-begin<=size && uint64_t(fields[i].bytes)<=size-(address-begin))direct.push_back(i);
            else staged.push_back(i);
        }
        std::sort(direct.begin(),direct.end(),[&](auto a,auto b) {
            return reinterpret_cast<uintptr_t>(fields[a].source.data_ptr())<reinterpret_cast<uintptr_t>(fields[b].source.data_ptr());
        });
        int64_t offset=cell.wire_header_bytes,direct_bytes=0,staged_bytes=0;
        std::vector<Interval> intervals;
        for(std::size_t first=0;first<direct.size();) {
            const auto address=reinterpret_cast<uintptr_t>(fields[direct[first]].source.data_ptr());
            auto end=address+fields[direct[first]].bytes;auto last=first+1;
            while(last<direct.size()) {
                const auto next=reinterpret_cast<uintptr_t>(fields[direct[last]].source.data_ptr());
                if(next<end || next-end>15 || ((next-address)&15))break;
                end=next+fields[direct[last]].bytes;++last;
            }
            offset=(offset+15)&~int64_t(15);
            const auto bytes=static_cast<int64_t>(end-address);
            if(bytes>cell.wire.numel()-offset)throw std::logic_error("MFQ direct RAM interval exceeds wire geometry");
            for(auto i=first;i<last;++i)fields[direct[i]].wire_offset=offset+
                static_cast<int64_t>(reinterpret_cast<uintptr_t>(fields[direct[i]].source.data_ptr())-address);
            intervals.push_back({reinterpret_cast<const uint8_t*>(address),bytes,offset});
            offset+=bytes;direct_bytes+=bytes;first=last;
        }
        const auto staged_start=staged.empty()?offset:(offset+15)&~int64_t(15);offset=staged_start;
        for(const auto i:staged) {
            offset=(offset+15)&~int64_t(15);auto& field=fields[i];
            if(field.bytes>cell.wire.numel()-offset || field.bytes>cache_->pipeline_host_stage_.numel()-offset)
                throw std::logic_error("MFQ exchanged RAM field exceeds wire geometry");
            std::memcpy(packed+offset,field.source.data_ptr(),static_cast<std::size_t>(field.bytes));
            field.wire_offset=offset;offset+=field.bytes;staged_bytes+=field.bytes;
        }
        if(!staged.empty() && offset>cell.wire.numel())throw std::logic_error("MFQ staged RAM transfer exceeds wire geometry");
        if(32+fields.size()*24>static_cast<std::size_t>(cell.wire_header_bytes))
            throw std::logic_error("MFQ direct RAM descriptors exceed route geometry");
        std::sort(fields.begin(),fields.end(),[](const auto& a,const auto& b){return a.wire_offset<b.wire_offset;});
        auto* words=reinterpret_cast<uint64_t*>(packed);
        words[0]=fields.size();words[1]=cell.wire_header_bytes;words[2]=offset;words[3]=0;
        int64_t payload_bytes=0;
        for(std::size_t i=0;i<fields.size();++i) {
            words[4+i*3]=reinterpret_cast<uint64_t>(fields[i].destination);
            words[5+i*3]=fields[i].wire_offset;words[6+i*3]=fields[i].bytes;payload_bytes+=fields[i].bytes;
        }
        auto* wire=cell.wire.data_ptr<uint8_t>();
        if(cell.two_stage && !cell.transfer_cache)for(const auto& field:fields) {
            const int index=cell.transfer_index->h<int32_t>()[field.expert];
            cell.transfer_views[index*3+field.projection].fields[field.field]=wire+field.wire_offset;
        }
        copy_transfer(cell,wire,packed,static_cast<std::size_t>(cell.wire_header_bytes));
        for(const auto& interval:intervals)copy_transfer(cell,wire+interval.offset,interval.source,static_cast<std::size_t>(interval.bytes));
        if(!staged.empty())copy_transfer(cell,wire+staged_start,packed+staged_start,static_cast<std::size_t>(offset-staged_start));
        const auto descriptor_bytes=upload_transfer_views(cell);
        const int64_t copies=1+intervals.size()+!staged.empty()+(descriptor_bytes>0);
        cache_->stats_.ram_pcie_bytes+=payload_bytes;
        cache_->stats_.pipeline_dma_copies+=copies;
        cache_->stats_.pipeline_direct_ram_bytes+=direct_bytes;
        cache_->stats_.pipeline_direct_ram_copies+=intervals.size();
        cache_->stats_.pipeline_staged_ram_bytes+=staged_bytes;
        if(cell.in_window) {
            cache_->stats_.pipeline_window_dma_copies+=copies;
            cache_->stats_.pipeline_window_dma_bytes+=descriptor_bytes+cell.wire_header_bytes+direct_bytes+
                (staged.empty()?0:offset-staged_start);
        }
        finish_transfer(cell,true);
        rollback.published=true;
    }
    void fetch(Cell& cell) {
        struct ClearPrefetchKeys {
            std::vector<mfq::MoeCacheKey>& keys;
            ~ClearPrefetchKeys(){keys.clear();}
        } clear_prefetch_keys{cache_->pipeline_prefetched_keys_};
        // The preceding call can finish CPU work before its weight DMA. Its
        // pinned source must remain immutable until that DMA has completed.
        if(cache_->transfer_ready_recorded_)MFQ_CUDA_CHECK(cudaEventSynchronize(cache_->transfer_ready_));
        cache_->pipeline_dma_leases_.clear();
        // Route publication on the same compute stream proves the preceding
        // layer finished reading these slots. DMA completion alone is weaker.
        for(const auto& held:cache_->pipeline_transfer_slots_)
            if(held.book->slot_for(held.key)==held.slot)held.book->clear_inflight(held.slot);
        cache_->pipeline_transfer_slots_.clear();
        if(!cell.two_stage)for(auto& item:cache_->pipeline_stages_)
            if(item.second->book)item.second->book=std::make_unique<mfq::MoeCacheSlotBook>(item.second->slots);
        if(!cell.mapped_copy && cache_->compute_done_recorded_)
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(cache_->weight_stream_,cache_->compute_done_,0));
        if(cell.wire.defined() && cache_->pipeline_ram_registered_bytes_) {
            fetch_registered(cell);return;
        }
        auto* packed=cache_->pipeline_host_stage_.data_ptr<uint8_t>();int64_t offset=cell.wire_header_bytes;
        struct Copy {void* dst;const void* src;int64_t bytes;std::size_t pitch=0,rows=1;};
        std::vector<Copy> copies;
        const auto* direct_option=std::getenv("MFQ_MOE_PREFILL_DIRECT_RAM");
        const bool direct_prefill=cell.tokens>8 && !cell.two_stage && !cell.wire.defined() &&
            cache_->pipeline_ram_registered_bytes_>0 && (!direct_option || std::atoi(direct_option)!=0);
        const auto ram_begin=cache_->pipeline_ram_complement_
            ? reinterpret_cast<std::uintptr_t>(cache_->pipeline_ram_complement_->data_ptr()) : std::uintptr_t(0);
        const auto ram_end=ram_begin+cache_->pipeline_ram_registered_bytes_;
        const auto registered=[&](const void* pointer,std::size_t bytes) {
            const auto address=reinterpret_cast<std::uintptr_t>(pointer);
            return direct_prefill && address>=ram_begin && address<=ram_end && bytes<=ram_end-address;
        };
        for(int projection=0;projection<3;++projection) {
            auto& source=*sources_[projection];std::vector<int> slots(source.cohorts_.size());
            struct Staged {int slot,expert;MoeHostExpertCache::Lease lease;std::vector<tb::Tensor> fields;};
            std::vector<std::vector<Staged>> staged(source.cohorts_.size());
            for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::GpuTransfer) {
                const int e=group.expert,c=source.expert_to_cohort_[e];auto& cohort=source.cohorts_[c];
                const int slot=slots[c]++;
                auto* stage=cell.two_stage?nullptr:cell.binding[projection].stages[c];
                if(stage && slot>=stage->slots)throw std::logic_error("MFQ transfer cohort exceeds source geometry");
                const int hot_slot=cohort.arena->book->slot_for({source.id_,c,e});
                // Split experts use resident GPU fields directly and upload
                // only their missing RAM projections.
                if(hot_slot>=0)continue;
                auto lease=source.quant_source_->acquire_expert(e);
                auto fields=moe_cache_fields(lease->weights);
                staged[c].push_back({slot,e,std::move(lease),std::move(fields)});
            }
            for(std::size_t c=0;c<staged.size();++c) {
                auto& experts=staged[c];const auto& cohort=source.cohorts_[c];
                const auto* stage=cell.two_stage?nullptr:cell.binding[projection].stages[c];
                for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                    const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
                    for(std::size_t first=0;first<experts.size();) {
                        if(direct_prefill && registered(experts[first].fields[f].data_ptr(),std::size_t(bytes))) {
                            const auto start=reinterpret_cast<std::uintptr_t>(experts[first].fields[f].data_ptr());
                            std::size_t last=first+1,pitch=0;
                            if(last<experts.size() && experts[last].slot==experts[first].slot+1) {
                                const auto next=reinterpret_cast<std::uintptr_t>(experts[last].fields[f].data_ptr());
                                if(next>start && next-start>=std::size_t(bytes) && next-start<=std::size_t(INT_MAX))
                                    pitch=next-start;
                            }
                            if(pitch)while(last<experts.size() && experts[last].slot==experts[last-1].slot+1 &&
                                    registered(experts[last].fields[f].data_ptr(),std::size_t(bytes)) &&
                                    reinterpret_cast<std::uintptr_t>(experts[last].fields[f].data_ptr())==start+(last-first)*pitch)++last;
                            const auto rows=last-first,total=std::size_t(bytes)*rows;
                            auto* destination=static_cast<uint8_t*>(stage->fields[f].data_ptr())+int64_t(experts[first].slot)*bytes;
                            if(rows>1 && pitch==std::size_t(bytes))copies.push_back({destination,experts[first].fields[f].data_ptr(),int64_t(total)});
                            else copies.push_back({destination,experts[first].fields[f].data_ptr(),bytes,pitch,rows});
                            cache_->stats_.ram_pcie_bytes+=total;
                            cache_->stats_.pipeline_direct_ram_bytes+=total;
                            ++cache_->stats_.pipeline_direct_ram_copies;
                            first=last;continue;
                        }
                        auto last=first+1;
                        while(last<experts.size() && experts[last].slot==experts[last-1].slot+1)++last;
                        const auto total=bytes*static_cast<int64_t>(last-first);
                        offset=(offset+15)&~int64_t(15);
                        if(total>static_cast<int64_t>(cache_->pipeline_host_stage_.numel())-offset)
                            throw std::logic_error("MFQ pinned stage geometry exceeded");
                        for(auto i=first;i<last;++i)std::memcpy(packed+offset+bytes*(i-first),
                            experts[i].fields[f].data_ptr(),static_cast<std::size_t>(bytes));
                        if(cell.two_stage)for(auto i=first;i<last;++i) {
                            const int index=cell.transfer_index->h<int32_t>()[experts[i].expert];
                            cell.transfer_views[index*3+projection].fields[f]=cell.wire.data_ptr<uint8_t>()+offset+bytes*(i-first);
                        }
                        auto* dst=stage?static_cast<uint8_t*>(stage->fields[f].data_ptr())+int64_t(experts[first].slot)*bytes:nullptr;
                        copies.push_back({dst,packed+offset,total});offset+=total;
                        if(direct_prefill)cache_->stats_.pipeline_staged_ram_bytes+=total;
                        cache_->stats_.ram_pcie_bytes+=total;first=last;
                    }
                }
                if(direct_prefill)for(auto& expert:experts)
                    cache_->pipeline_dma_leases_.push_back(std::move(expert.lease));
            }
        }
        const auto descriptor_bytes=upload_transfer_views(cell);
        const auto submissions=(cell.wire.defined() ? 1 : static_cast<int64_t>(copies.size()))+(descriptor_bytes>0);
        cache_->stats_.pipeline_dma_copies+=submissions;
        if(cell.in_window) {
            cache_->stats_.pipeline_window_dma_copies+=submissions;
            if(cell.wire.defined())cache_->stats_.pipeline_window_dma_bytes+=offset+descriptor_bytes;
            else for(const auto& copy:copies)cache_->stats_.pipeline_window_dma_bytes+=copy.bytes;
        }
        // The GPU's plan becomes visible before DMA submission and CPU jobs.
        if(cell.wire.defined()) {
            if(offset>cell.wire.numel() || 32+copies.size()*24>static_cast<std::size_t>(cell.wire_header_bytes))
                throw std::logic_error("MFQ packed DMA exceeded its real route geometry");
            auto* words=reinterpret_cast<uint64_t*>(packed);
            words[0]=copies.size();words[1]=cell.wire_header_bytes;words[2]=offset;words[3]=0;
            for(std::size_t i=0;i<copies.size();++i) {
                words[4+i*3]=reinterpret_cast<uint64_t>(copies[i].dst);
                words[5+i*3]=static_cast<const uint8_t*>(copies[i].src)-packed;
                words[6+i*3]=copies[i].bytes;
            }
            copy_transfer(cell,cell.wire.data_ptr(),packed,static_cast<std::size_t>(offset));
        } else for(const auto& copy:copies)copy_transfer(cell,copy.dst,copy.src,static_cast<std::size_t>(copy.bytes),copy.pitch,copy.rows);
        finish_transfer(cell,!copies.empty() || cell.wire.defined());
    }
    void cpu(Cell& cell) {
        using Clock=std::chrono::steady_clock;
        const auto ns=[](Clock::time_point begin,Clock::time_point end) {
            return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        };
        struct ProjectionTiming {
            std::atomic<std::uint64_t> work_ns{0},callbacks{0},rows{0},activation_ns{0};
        };
        MoeExpertCache::CpuCallSample sample;
        std::unique_ptr<ProjectionTiming[]> timing;
        auto started=cache_->pipeline_cpu_profile_enabled_ ? Clock::now() : Clock::time_point{};
        if(cache_->pipeline_cpu_profile_enabled_)++cell.cpu_profile_calls;
        cell.jobs.clear();int offset=0;
        for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::Cpu) {
            CpuJob job;job.expert=group.expert;job.offset=offset;job.count=static_cast<int>(group.positions.size());
            for(int i=0;i<3;++i){job.leases[i]=sources_[i]->quant_source_->acquire_expert(group.expert);job.rows[i]=rows_plan(job.leases[i]->weights);}
            offset+=job.count;cell.jobs.push_back(std::move(job));
        }
        cache_->record_hybrid(static_cast<int>(cell.jobs.size()),static_cast<int>(cell.plan.groups.size()-cell.jobs.size()));
        for(const auto& source:sources_)source->quant_source_->record_cpu_projections(cell.jobs.size());
        for(const auto& group:cell.plan.groups)cache_->stats_.ram_pcie_experts+=group.kind==mfq::MoeDispatchKind::GpuTransfer;
        if(cell.jobs.empty()){publish(cell.flags.h<uint32_t>()+48);return;}
        cell.cpu_input.resize(std::size_t(offset)*width_);cell.g.resize(std::size_t(offset)*ff_);
        cell.u.resize(std::size_t(offset)*ff_);cell.hidden.resize(std::size_t(offset)*ff_);cell.y.resize(std::size_t(offset)*width_);
        for(const auto& job:cell.jobs) {
            const auto& positions=std::find_if(cell.plan.groups.begin(),cell.plan.groups.end(),[&](const auto& g){return g.expert==job.expert;})->positions;
            for(int row=0;row<job.count;++row)std::memcpy(cell.cpu_input.data()+std::size_t(job.offset+row)*width_,
                cell.x.h<float>()+(positions[row]/cell.routes)*width_,width_*sizeof(float));
        }
        if(cache_->pipeline_cpu_profile_enabled_) {
            // Only auxiliary storage failure disables tracing. Lease, plan,
            // projection and result failures still propagate to the caller.
            try {
                if(cache_->pipeline_cpu_profile_fail_alloc_)throw std::bad_alloc();
                timing=std::make_unique<ProjectionTiming[]>(cell.jobs.size()*3);
                sample.projections.resize(cell.jobs.size()*3);
            } catch(const std::bad_alloc&) {
                timing.reset();cache_->pipeline_cpu_profile_enabled_=false;
                cache_->pipeline_cpu_profile_dropped_=true;
            }
        }
        const auto prepared=timing ? Clock::now() : Clock::time_point{};
        const auto projection=[&](std::size_t index,int p,const float* input,int stride,
                float* output,int out,int first,int last) {
            const auto& job=cell.jobs[index];
            if(!timing) { job.rows[p].run(input,job.count,stride,output,out,first,last);return; }
            const auto before=Clock::now();
            job.rows[p].run(input,job.count,stride,output,out,first,last);
            auto& t=timing[index*3+p];
            t.work_ns.fetch_add(ns(before,Clock::now()),std::memory_order_relaxed);
            t.callbacks.fetch_add(1,std::memory_order_relaxed);
            t.rows.fetch_add(last-first,std::memory_order_relaxed);
        };
        mfq::cpu::expert_pool().rows(int64_t(cell.jobs.size())*ff_,[&](int64_t begin,int64_t end) {
            while(begin<end) {
                const auto index=static_cast<std::size_t>(begin/ff_);
                const auto& job=cell.jobs[index];const int first=static_cast<int>(begin%ff_);
                const int last=static_cast<int>(std::min<int64_t>(ff_,first+end-begin));
                const auto* input=cell.cpu_input.data()+std::size_t(job.offset)*width_;
                projection(index,0,input,width_,cell.g.data()+std::size_t(job.offset)*ff_,ff_,first,last);
                projection(index,1,input,width_,cell.u.data()+std::size_t(job.offset)*ff_,ff_,first,last);
                const auto activation_started=timing ? Clock::now() : Clock::time_point{};
                // Every row chunk owns both projections and its hidden values.
                // Keep each original F16 operation before the Down dispatch.
                for(int row=0;row<job.count;++row)for(int col=first;col<last;++col) {
                    const auto i=std::size_t(job.offset+row)*ff_+col;
                    const mfq_half g(cell.g[i]),u(cell.u[i]);
                    const mfq_half sigmoid(1.0/(1.0+std::exp(-double(float(g)))));
                    const mfq_half first(float(g)*float(sigmoid));
                    cell.hidden[i]=float(mfq_half(float(first)*float(u)));
                }
                if(timing)timing[index*3].activation_ns.fetch_add(
                    ns(activation_started,Clock::now()),std::memory_order_relaxed);
                begin+=last-first;
            }
        },timing ? &sample.gate_up : nullptr);
        mfq::cpu::expert_pool().rows(int64_t(cell.jobs.size())*width_,[&](int64_t begin,int64_t end) {
            while(begin<end) {
                const auto index=static_cast<std::size_t>(begin/width_);
                const auto& job=cell.jobs[index];const int first=static_cast<int>(begin%width_);
                const int last=static_cast<int>(std::min<int64_t>(width_,first+end-begin));
                projection(index,2,cell.hidden.data()+std::size_t(job.offset)*ff_,ff_,
                    cell.y.data()+std::size_t(job.offset)*width_,width_,first,last);begin+=last-first;
            }
        },timing ? &sample.down : nullptr);
        const auto scatter_started=timing ? Clock::now() : Clock::time_point{};
        for(const auto& job:cell.jobs) {
            const auto& positions=std::find_if(cell.plan.groups.begin(),cell.plan.groups.end(),[&](const auto& g){return g.expert==job.expert;})->positions;
            for(int row=0;row<job.count;++row)for(int col=0;col<width_;++col)
                cell.cpu.h<mfq_half>()[positions[row]*width_+col]=mfq_half(cell.y[std::size_t(job.offset+row)*width_+col]);
        }
        publish(cell.flags.h<uint32_t>()+48);
        if(timing) {
            const auto finished=Clock::now();
            sample.serve=cache_->stats_.pipeline_serves;sample.cell_call=cell.cpu_profile_calls;
            sample.layer=sources_[0]->layer_id_;sample.tokens=cell.tokens;sample.window=cell.in_window;
            sample.experts=static_cast<int>(cell.jobs.size());sample.positions=offset;
            sample.prepare_ns=ns(started,prepared);sample.activation_inside_gate_up=true;
            sample.scatter_ns=ns(scatter_started,finished);sample.total_ns=ns(started,finished);
            for(std::size_t i=0;i<cell.jobs.size();++i)for(int p=0;p<3;++p) {
                const auto& job=cell.jobs[i];const auto& w=job.leases[p]->weights;
                const auto& t=timing[i*3+p];auto& r=sample.projections[i*3+p];
                sample.activation_work_ns+=t.activation_ns.load();
                r.expert=job.expert;r.projection=p;r.family=static_cast<int>(w.family);
                r.format=static_cast<int>(w.family==MixedMoeFamily::Nvq ? w.nvq.kernel_format : w.nint.format_version);
                r.bits=static_cast<int>(w.family==MixedMoeFamily::Nvq ? w.nvq.sub_bits : w.nint.bits);
                r.group_size=static_cast<int>(w.family==MixedMoeFamily::Nvq ? w.nvq.gs : w.nint.gs);
                r.positions=job.count;r.width=job.rows[p].width;r.outputs=job.rows[p].outputs;
                r.work_ns=t.work_ns.load();r.callbacks=t.callbacks.load();r.rows=t.rows.load();
            }
            try {cache_->pipeline_cpu_profiles_.push_back(std::move(sample));}
            catch(const std::bad_alloc&) {
                cache_->pipeline_cpu_profile_enabled_=false;cache_->pipeline_cpu_profile_dropped_=true;
            }
        }
        cell.jobs.clear();
    }
    void trace_serve(const Cell& cell,const char* phase,bool force=false) const noexcept {
        const auto serves=cache_->stats_.pipeline_window_serves;
        if(!force && (!trace_serve_ || (cell.in_window && serves>=96 && serves%(48*64)>=48)))return;
        const auto* flags=static_cast<const volatile uint32_t*>(cell.flags.h<uint32_t>());
        std::fprintf(stderr,"moe_serve layer=%d tokens=%d window=%d serve=%llu window_serve=%llu phase=%s flags=%u,%u,%u,%u,%u,%u\n",
            sources_[0]->layer_id_,cell.tokens,int(cell.in_window),
            static_cast<unsigned long long>(cache_->stats_.pipeline_serves),
            static_cast<unsigned long long>(serves),phase,
            flags[0],flags[16],flags[32],flags[48],flags[64],flags[80]);
        std::fflush(stderr);
    }
    void serve(Cell& cell) {
        auto& stats=cache_->stats_;++stats.pipeline_serves;
        stats.pipeline_two_stage_serves+=cell.two_stage;
        stats.pipeline_window_input_view_serves+=cell.window_input_views;
        stats.pipeline_phased_transfer_serves+=cell.phased_transfer;
        const auto route_started=stats.pipeline_route_wait_ns,plan_started=stats.pipeline_plan_ns;
        const auto fetch_before=stats.pipeline_fetch_ns,cpu_before=stats.pipeline_cpu_ns;
        const auto time=[](int64_t& total,const auto& fn) {
            struct Sample {
                int64_t& total;
                std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
                ~Sample(){total+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-started).count();}
            } sample{total};
            fn();
        };
        trace_serve(cell,"route_wait");
        time(stats.pipeline_route_wait_ns,[&]{wait_publication(cell);});
        trace_serve(cell,"plan");
        time(stats.pipeline_plan_ns,[&]{plan(cell);});
        trace_serve(cell,"fetch");
        const auto transferred=stats.ram_pcie_bytes,fetch_started=stats.pipeline_fetch_ns;
        if(cache_->pipeline_dma_profile_enabled_) {
            if(cell.dma_timing) {
                auto& timing=*cell.dma_timing;timing.started=false;timing.sample={};
                auto& sample=timing.sample;
                sample.layer=sources_[0]->layer_id_;sample.tokens=cell.tokens;sample.entries=cell.entries;
                sample.window=cell.in_window;sample.serve=stats.pipeline_serves;
                sample.route_wait_ns=stats.pipeline_route_wait_ns-route_started;
                sample.plan_ns=stats.pipeline_plan_ns-plan_started;
            } else ++cache_->pipeline_dma_profile_skipped_;
        }
        time(stats.pipeline_fetch_ns,[&] {
            if(cache_->pipeline_dma_profile_enabled_ && cell.dma_timing)
                cell.dma_timing->fetch_started=std::chrono::steady_clock::now();
            fetch(cell);
        });
        trace_serve(cell,"cpu");
        if(cache_->pipeline_dma_profile_enabled_ && cell.dma_timing) {
            auto& sample=cell.dma_timing->sample;
            sample.payload_bytes=stats.ram_pcie_bytes-transferred;
            sample.fetch_ns=stats.pipeline_fetch_ns-fetch_started;
        }
        if(stats.ram_pcie_bytes>transferred) {
            stage_bytes_+=stats.ram_pcie_bytes-transferred;stage_ns_+=stats.pipeline_fetch_ns-fetch_started;
        }
        std::size_t work=0;for(const auto& group:cell.plan.groups)
            if(group.kind==mfq::MoeDispatchKind::Cpu)work+=group.positions.size();
        const auto cpu_started=stats.pipeline_cpu_ns;
        if(cell.calibration_expert>=0) {
            const auto prepared=std::chrono::steady_clock::now();
            const auto found=std::find_if(cell.plan.groups.begin(),cell.plan.groups.end(),
                [&](const auto& group){return group.expert==cell.calibration_expert;});
            if(found==cell.plan.groups.end() || found->kind==mfq::MoeDispatchKind::Cpu)
                throw std::logic_error("background CPU calibration changed production dispatch");
            std::array<MoeHostExpertCache::Lease,3> leases;
            for(int p=0;p<3;++p)leases[p]=sources_[p]->quant_source_->acquire_expert(found->expert);
            std::vector<float> input(found->positions.size()*width_);
            for(size_t row=0;row<found->positions.size();++row)
                std::memcpy(input.data()+row*width_,cell.x.h<float>()+
                    (found->positions[row]/cell.routes)*width_,width_*sizeof(float));
            const auto preparation_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-prepared).count();
            auto work=cpu_calibration_work(cpu_cost_key(*found),std::move(leases),std::move(input),
                width_,ff_,int(found->positions.size()),double(preparation_ns));
            cache_->pipeline_cpu_calibration_.start(std::move(work));
        }
        time(stats.pipeline_cpu_ns,[&]{cpu(cell);});
        trace_serve(cell,"complete");
        if(work){
            const auto elapsed=stats.pipeline_cpu_ns-cpu_started;
            cpu_ns_+=elapsed;cpu_work_+=work;
            if(cache_->pipeline_shared_cpu_cost_) {
                mfq::MoeCpuCostModel::Key key{};std::size_t groups=0;bool homogeneous=true;
                for(const auto& group:cell.plan.groups)if(group.kind==mfq::MoeDispatchKind::Cpu) {
                    const auto candidate=cpu_cost_key(group);
                    if(!groups)key=candidate;else homogeneous&=candidate==key;
                    ++groups;
                }
                // Shared pool walltime cannot identify independent rates for
                // heterogeneous jobs; retain only unambiguous observations.
                if(homogeneous)cache_->pipeline_cpu_cost_.observe(key,double(elapsed),groups);
            }
        }
        if(cell.in_window) {
            ++stats.pipeline_window_serves;
            stats.pipeline_window_route_ns+=stats.pipeline_route_wait_ns-route_started;
            stats.pipeline_window_plan_ns+=stats.pipeline_plan_ns-plan_started;
            stats.pipeline_window_fetch_ns+=stats.pipeline_fetch_ns-fetch_before;
            stats.pipeline_window_cpu_ns+=stats.pipeline_cpu_ns-cpu_before;
        }
    }
    void observe_gpu(Cell& cell) {
        // Window finish runs after the complete GPU graph. Join at most its
        // final background sample before any RAM lease can be exchanged.
        if(cell.in_window)cache_->pipeline_cpu_calibration_.collect(cache_->pipeline_cpu_cost_,true);
        if(cell.skip_gpu_timing)return;
        float hot=0,transfer=0;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&hot,cell.timing.points[0],cell.timing.points[1]));
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&transfer,cell.timing.points[2],cell.timing.points[3]));
        const bool pure_gpu=!cell.two_stage || std::all_of(cell.plan.groups.begin(),cell.plan.groups.end(),
            [](const auto& group){return group.kind!=mfq::MoeDispatchKind::Cpu;});
        if(cell.tokens<=8 && pure_gpu){gpu_ns_+=(hot+transfer)*1e6;++gpu_samples_;}
        cache_->stats_.pipeline_gpu_ns+=static_cast<int64_t>((hot+transfer)*1e6);
        if(cell.in_window)cache_->stats_.pipeline_window_gpu_ns+=static_cast<int64_t>((hot+transfer)*1e6);
        if(cache_->pipeline_dma_profile_enabled_ && cell.dma_timing && cell.dma_timing->started) {
            auto& timing=*cell.dma_timing;auto& p=timing.sample;
            // The third event follows the stream flag write or callback fallback,
            // so it bounds notification completion.
            // Stream intervals include enqueue/queue gaps, not just copy-engine
            // execution. Synchronization and all extra events are opt-in only.
            MFQ_CUDA_CHECK(cudaEventSynchronize(timing.points[2]));
            const auto elapsed=[](cudaEvent_t begin,cudaEvent_t end) {
                float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
                return static_cast<std::int64_t>(std::llround(double(ms)*1e6));
            };
            p.copy_ns=elapsed(timing.points[0],timing.points[1]);
            p.notice_ns=elapsed(timing.points[1],timing.points[2]);
            p.hot_ns=static_cast<std::int64_t>(std::llround(double(hot)*1e6));
            p.copy_begin_after_hot_ns=elapsed(cell.timing.points[0],timing.points[0]);
            p.copy_end_after_hot_ns=elapsed(cell.timing.points[0],timing.points[1]);
            p.notice_end_after_hot_ns=elapsed(cell.timing.points[0],timing.points[2]);
            p.scatter_end_after_hot_ns=elapsed(cell.timing.points[0],cell.timing.points[2]);
            try {cache_->pipeline_dma_profiles_.push_back(p);}
            catch(const std::bad_alloc&) {
                cache_->pipeline_dma_profile_enabled_=false;cache_->pipeline_dma_profile_dropped_=true;
            }
            timing.started=false;
        }
    }
public:
    void prefetch_host(const std::vector<int32_t>& predictions,Mapped& flags) {
        struct Field {void* destination;tb::Tensor source;int64_t bytes,offset=0;};
        std::vector<Field> fields;
        const auto lease_begin=cache_->pipeline_dma_leases_.size();
        const auto slot_begin=cache_->pipeline_transfer_slots_.size();
        const auto key_begin=cache_->pipeline_prefetched_keys_.size();
        fields.reserve(predictions.size()*3*7);
        cache_->pipeline_dma_leases_.reserve(lease_begin+predictions.size()*3);
        cache_->pipeline_transfer_slots_.reserve(slot_begin+predictions.size()*3);
        cache_->pipeline_prefetched_keys_.reserve(key_begin+predictions.size()*3);
        struct Rollback {
            MoeExpertCache& cache;std::size_t leases,slots,keys;bool committed=false;
            ~Rollback() {
                if(committed)return;
                (void)cudaStreamSynchronize(cache.weight_stream_);
                for(auto i=slots;i<cache.pipeline_transfer_slots_.size();++i) {
                    const auto& held=cache.pipeline_transfer_slots_[i];
                    held.book->discard(held.key,held.slot,held.generation);
                }
                cache.pipeline_transfer_slots_.resize(slots);
                cache.pipeline_dma_leases_.resize(leases);
                cache.pipeline_prefetched_keys_.resize(keys);
            }
        } rollback{*cache_,lease_begin,slot_begin,key_begin};
        const int64_t capacity=cache_->pipeline_prefetch_host_stage_.numel();
        const int64_t header_bytes=(32+predictions.size()*3*7*24+15)&~int64_t(15);
        int64_t estimated=header_bytes;
        std::vector<int32_t> seen;
        for(const int e:predictions) {
            if(e<0 || e>=experts_)throw std::runtime_error("MFQ prefetch expert outside router geometry");
            if(std::find(seen.begin(),seen.end(),e)!=seen.end())continue;
            seen.push_back(e);bool copied=false;
            for(int p=0;p<3;++p) {
                auto& source=*sources_[p];const int c=source.expert_to_cohort_[e];
                auto& cohort=source.cohorts_[c];if(source.quant_source_->gpu_resident(e))continue;
                auto& arena=stage(p,cohort);auto* book=arena.book.get();const mfq::MoeCacheKey key{source.id_,c,e};
                if(book->slot_for(key)>=0)continue;
                bool available=false;for(int slot=0;slot<book->capacity();++slot)available|=!book->inflight(slot);
                if(!available){++cache_->stats_.pipeline_prefetch_busy_skips;continue;}
                const auto size=static_cast<int64_t>(source.bytes_for_expert(e));
                if(size+int64_t(cohort.bytes_per_expert.size())*15>capacity-estimated)continue;
                auto lease=source.quant_source_->acquire_expert(e);auto values=moe_cache_fields(lease->weights);
                if(values.size()!=cohort.bytes_per_expert.size())throw std::logic_error("prefetch RAM field count changed");
                const auto held=book->acquire(key);
                cache_->pipeline_transfer_slots_.push_back({book,key,held.slot,held.generation});book->mark_inflight(held.slot);
                cache_->pipeline_prefetched_keys_.push_back(key);
                for(std::size_t f=0;f<values.size();++f) {
                    const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
                    if(tensor_nbytes(values[f])!=bytes)throw std::logic_error("prefetch RAM field geometry changed");
                    fields.push_back({static_cast<uint8_t*>(arena.fields[f].data_ptr())+int64_t(held.slot)*bytes,std::move(values[f]),bytes});
                }
                estimated+=size+int64_t(values.size())*15;
                cache_->pipeline_dma_leases_.push_back(std::move(lease));copied=true;
            }
            cache_->stats_.pipeline_prefetch_experts+=copied;
        }
        if(fields.empty()) {
            publish(flags.h<uint32_t>()+8);publish(flags.h<uint32_t>()+4);
            rollback.committed=true;return;
        }
        auto* host=cache_->pipeline_prefetch_host_stage_.data_ptr<uint8_t>();
        auto* wire=cache_->pipeline_prefetch_wire_gpu_.data_ptr<uint8_t>();
        const auto ram_begin=cache_->pipeline_ram_complement_?reinterpret_cast<uintptr_t>(cache_->pipeline_ram_complement_->data_ptr()):uintptr_t(0);
        const auto ram_bytes=static_cast<uint64_t>(cache_->pipeline_ram_registered_bytes_);
        std::vector<std::size_t> direct,staged;
        for(std::size_t i=0;i<fields.size();++i) {
            const auto address=reinterpret_cast<uintptr_t>(fields[i].source.data_ptr());
            (address>=ram_begin && address-ram_begin<=ram_bytes && uint64_t(fields[i].bytes)<=ram_bytes-(address-ram_begin)?direct:staged).push_back(i);
        }
        std::sort(direct.begin(),direct.end(),[&](auto a,auto b){return reinterpret_cast<uintptr_t>(fields[a].source.data_ptr())<reinterpret_cast<uintptr_t>(fields[b].source.data_ptr());});
        struct Interval {const void* source;int64_t bytes,offset;};std::vector<Interval> intervals;
        int64_t offset=header_bytes,direct_bytes=0,staged_bytes=0;
        for(std::size_t first=0;first<direct.size();) {
            const auto begin=reinterpret_cast<uintptr_t>(fields[direct[first]].source.data_ptr());
            auto end=begin+fields[direct[first]].bytes;auto last=first+1;
            while(last<direct.size()) {
                const auto next=reinterpret_cast<uintptr_t>(fields[direct[last]].source.data_ptr());
                if(next<end || next-end>15 || ((next-begin)&15))break;
                end=next+fields[direct[last]].bytes;++last;
            }
            offset=(offset+15)&~int64_t(15);const auto count=static_cast<int64_t>(end-begin);
            if(count>capacity-offset)throw std::logic_error("prefetch wire interval capacity exceeded");
            for(auto i=first;i<last;++i) {
                auto& field=fields[direct[i]];field.offset=offset+static_cast<int64_t>(reinterpret_cast<uintptr_t>(field.source.data_ptr())-begin);
                direct_bytes+=field.bytes;
            }
            intervals.push_back({reinterpret_cast<const void*>(begin),count,offset});offset+=count;first=last;
        }
        const auto staged_begin=(offset+15)&~int64_t(15);offset=staged_begin;
        for(const auto i:staged) {
            offset=(offset+15)&~int64_t(15);auto& field=fields[i];
            if(field.bytes>capacity-offset)throw std::logic_error("prefetch staged field capacity exceeded");
            field.offset=offset;std::memcpy(host+offset,field.source.data_ptr(),field.bytes);offset+=field.bytes;staged_bytes+=field.bytes;
        }
        std::sort(fields.begin(),fields.end(),[](const auto& a,const auto& b){return a.offset<b.offset;});
        auto* words=reinterpret_cast<uint64_t*>(host);words[0]=fields.size();words[1]=header_bytes;words[2]=offset;words[3]=0;
        int64_t payload=0;
        for(std::size_t i=0;i<fields.size();++i) {
            words[4+i*3]=reinterpret_cast<uint64_t>(fields[i].destination);words[5+i*3]=fields[i].offset;words[6+i*3]=fields[i].bytes;payload+=fields[i].bytes;
        }
        MFQ_CUDA_CHECK(cudaMemcpyAsync(wire,host,header_bytes,cudaMemcpyHostToDevice,cache_->weight_stream_));
        for(const auto& interval:intervals)MFQ_CUDA_CHECK(cudaMemcpyAsync(wire+interval.offset,interval.source,interval.bytes,cudaMemcpyHostToDevice,cache_->weight_stream_));
        if(!staged.empty())MFQ_CUDA_CHECK(cudaMemcpyAsync(wire+staged_begin,host+staged_begin,offset-staged_begin,cudaMemcpyHostToDevice,cache_->weight_stream_));
        MFQ_CUDA_CHECK(cudaLaunchHostFunc(cache_->weight_stream_,[](void* p){publish(static_cast<uint32_t*>(p));},flags.h<uint32_t>()+4));
        MFQ_CUDA_CHECK(cudaEventRecord(cache_->transfer_ready_,cache_->weight_stream_));cache_->transfer_ready_recorded_=true;
        cache_->stats_.pipeline_prefetch_bytes+=payload;cache_->stats_.ram_pcie_bytes+=payload;
        cache_->stats_.pipeline_direct_ram_bytes+=direct_bytes;cache_->stats_.pipeline_staged_ram_bytes+=staged_bytes;
        cache_->stats_.pipeline_window_dma_bytes+=offset;
        cache_->stats_.pipeline_window_dma_copies+=1+intervals.size()+!staged.empty();
        mfq::cuda::check_stream_progress(cache_->weight_stream_);
        rollback.committed=true;
    }
    mfq::cuda::DecodeWindow::Task prediction(const tb::Tensor& ids) {
        if(!mfq::cuda::DecodeWindow::recording() || ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>2 ||
            ids.scalar_type()!=tb::kInt32 || !ids.is_cuda() || !cache_->pipeline_transfer_cache_ || cache_->pipeline_mapped_copy_)
            throw std::invalid_argument("MFQ prefetch requires a single decode row, DMA and transfer caching");
        constexpr int64_t capacity=16*1024*1024;
        if(!cache_->pipeline_prefetch_host_stage_.defined()) {
            cache_->pipeline_prefetch_wire_gpu_=tb::empty({capacity},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
            cache_->pipeline_prefetch_host_stage_=tb::empty({capacity},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        }
        for(int p=0;p<3;++p)for(auto& cohort:sources_[p]->cohorts_) {
            auto& arena=stage(p,cohort);if(!arena.book)arena.book=std::make_unique<mfq::MoeCacheSlotBook>(arena.slots);
        }
        const auto* window=mfq::cuda::DecodeWindow::recording();
        auto& entry=prediction_cells_[window];if(!entry)entry=std::make_shared<PredictionCell>();
        const int count=static_cast<int>(ids.numel());auto prediction=entry;
        auto* mapped=&prediction->flags;
        mfq::cuda::moe_publish_routes(nullptr,ids.data_ptr<int32_t>(),nullptr,mapped->d<int32_t>()+16,mapped->d<uint32_t>(),0,count,mfq_current_cuda_stream());
        auto owner=shared_from_this();const auto stream=mfq_current_cuda_stream();
        return {prediction.get(),[prediction]{std::memset(prediction->flags.host,0,prediction->flags.bytes);},
            [owner,prediction,mapped,count,stream] {
                mfq::cuda::wait_route_publication(mapped->h<uint32_t>(),stream);
                owner->prefetch_host(std::vector<int32_t>(mapped->h<int32_t>()+16,mapped->h<int32_t>()+16+count),*mapped);
            },{}, [prediction] {
                publish(prediction->flags.h<uint32_t>()+8);publish(prediction->flags.h<uint32_t>()+4);
            },[owner,window] {
                (void)cudaStreamSynchronize(owner->cache_->weight_stream_);
                owner->prediction_cells_.erase(window);
            }};
    }
    ~MoeFfnPipeline() {
        // A host error releases every graph wait. Drain both producers before
        // destroying mapped flags that a graph or DMA callback still uses.
        if(stream_)(void)cudaStreamSynchronize(stream_);
        if(cache_->weight_stream_)(void)cudaStreamSynchronize(cache_->weight_stream_);
    }
    MoeFfnPipeline(std::array<std::shared_ptr<MoeCachedSource>,3> sources,MoeFfnShared shared,MoeFfnDispatchObserver observer,
            MoeFfnSharedWeights shared_weights)
        :sources_(std::move(sources)),cache_(sources_[0]->cache_),shared_(std::move(shared)),shared_weights_(std::move(shared_weights)),observer_(std::move(observer)),
        width_(sources_[0]->cpu_->neuron_len),ff_(sources_[0]->cpu_->out_per_expert),experts_(sources_[0]->n_experts()) {
        const auto* trace=std::getenv("MFQ_TRACE_MOE_SERVE");
        trace_serve_=trace && trace[0] && trace[0]!='0';
        for(const auto& s:sources_)if(!s->quant_source_ || s->cache_!=cache_ || s->n_experts()!=experts_)
            throw std::invalid_argument("MFQ FFN sources must share one expert cache");
        if(sources_[1]->cpu_->neuron_len!=width_ || sources_[1]->cpu_->out_per_expert!=ff_ ||
            sources_[2]->cpu_->neuron_len!=ff_ || sources_[2]->cpu_->out_per_expert!=width_)
            throw std::invalid_argument("MFQ FFN projection geometry mismatch");
        const std::array<int,3> bundle{sources_[0]->id_,sources_[1]->id_,sources_[2]->id_};
        if(bundle[0]==bundle[1] || bundle[0]==bundle[2] || bundle[1]==bundle[2])
            throw std::invalid_argument("MFQ FFN requires distinct projection sources");
        cpu_cost_keys_.resize(experts_);
        for(int e=0;e<experts_;++e)for(int p=0;p<3;++p) {
            const auto& source=*sources_[p];
            cpu_cost_keys_[e][p]=reinterpret_cast<std::uintptr_t>(
                source.cohorts_[source.expert_to_cohort_[e]].arena);
        }
        if(std::find(cache_->pipeline_bundles_.begin(),cache_->pipeline_bundles_.end(),bundle)==cache_->pipeline_bundles_.end()) {
            if(cache_->moe_residency_)throw std::logic_error("MFQ cannot add an expert bundle after scheduling starts");
            cache_->pipeline_bundles_.push_back(bundle);
        }
    }
    tb::Tensor forward(CudaExecutionContext& execution,const tb::Tensor& input,const tb::Tensor& ids,const tb::Tensor& weights) {
        if(failed_)throw std::runtime_error("MFQ FFN session requires recovery after its previous error");
        if(!cache_->complete_residency())throw std::runtime_error("MFQ FFN needs completed startup residency");
        const auto current=mfq_current_cuda_stream();
        if(stream_ && stream_!=current)throw std::runtime_error("MFQ FFN session changed its CUDA stream");
        stream_=current;
        auto* window=mfq::cuda::DecodeWindow::recording();
        const int tokens=static_cast<int>(ids.size(0)),routes=static_cast<int>(ids.size(1));
        if(!window && tokens<=8)release_moe_prefill_buffers(cache_);
        if(input.dim()!=2 || input.size(0)!=tokens || input.size(1)!=width_ || !input.is_cuda() ||
            input.scalar_type()!=tb::kFloat16 || ids.dim()!=2 || ids.scalar_type()!=tb::kInt32 || !ids.is_cuda() ||
            weights.sizes().vec()!=ids.sizes().vec())throw std::invalid_argument("MFQ FFN input/route geometry mismatch");
        // Keep temporary pages reusable across every prefill path. The existing
        // allocation limits still apply, and decode restores the old threshold.
        if(!window && tokens>8)retain_prefill_pool(ids);
        const auto* layer_prefill=std::getenv("MFQ_MOE_PREFILL_LAYER");
        if(!window && tokens>8 && layer_prefill && std::atoi(layer_prefill)!=0 &&
                cache_->pipeline_ram_registered_bytes_>0 && execution.kl_mmq.mode==KlMmqMode::Default) {
            try {return whole_layer_prefill(execution,input,ids,weights);}
            catch(...) {failed_=true;throw;}
        }
        if(!cache_->moe_residency_)cache_->moe_residency_=std::make_unique<MoeResidencyManager>(cache_);
        if(!window)cache_->moe_residency_->before_layer(sources_[0]->layer_id_);
        const auto* prefill_groups=std::getenv("MFQ_MOE_PREFILL_GROUPS");
        if(!window && tokens>8 && prefill_groups && std::atoi(prefill_groups)!=0 &&
                cache_->pipeline_ram_registered_bytes_>0 && cache_->pipeline_dispatch_replay_.empty() &&
                execution.kl_mmq.mode==KlMmqMode::Default) {
            try {return streamed_prefill(execution,input,ids,weights);}
            catch(...) {failed_=true;throw;}
        }
        auto& cell=cells_[{tokens,routes,window}];
        if(!cell) {
            const auto* prefill_graph=std::getenv("MFQ_MOE_PREFILL_GRAPH");
            const bool eager=!window && tokens>8 && prefill_graph && std::atoi(prefill_graph)==0;
            cell=std::make_unique<Cell>(tokens,routes,width_,experts_,stream_,eager);
            initialize(*cell,execution,window!=nullptr);
        }
        if(window) {
            auto* captured=cell.get();
            window->enroll({captured,
                [this,captured,owner=shared_from_this()] {
                    if(failed_)throw std::runtime_error("MFQ FFN session requires recovery");
                    cache_->moe_residency_->before_layer(sources_[0]->layer_id_);
                    std::memset(captured->flags.host,0,captured->flags.bytes);
                },
                [this,captured] {
                    serve(*captured);
                    if(cache_->moe_residency_->prepare_during_window()) {
                        const std::vector<int32_t> routed(captured->host_ids.h<int32_t>(),captured->host_ids.h<int32_t>()+captured->entries);
                        cache_->moe_residency_->after_layer(sources_[0]->layer_id_,routed,captured->tokens,true);
                    }
                },
                [this,captured] {
                    observe_gpu(*captured);
                    cache_->pipeline_dma_leases_.clear();
                    if(cache_->moe_residency_->prepare_during_window())cache_->moe_residency_->finish_window(sources_[0]->layer_id_);
                    else {
                        cache_->record_compute_use();
                        const std::vector<int32_t> routed(captured->host_ids.h<int32_t>(),captured->host_ids.h<int32_t>()+captured->entries);
                        cache_->moe_residency_->after_layer(sources_[0]->layer_id_,routed,captured->tokens);
                    }
                },
                [this,captured] {
                    publish(captured->flags.h<uint32_t>()+64);publish(captured->flags.h<uint32_t>()+80);for(int i=1;i<4;++i)publish(captured->flags.h<uint32_t>()+16*i);
                    cache_->moe_residency_->finish_window(sources_[0]->layer_id_);
                },
                [this,key=std::make_tuple(tokens,routes,static_cast<const void*>(window))] {
                    (void)cudaStreamSynchronize(cache_->weight_stream_);cells_.erase(key);
                },[captured] {captured->output={};}});
            cell->window_input_views=cell->two_stage && mfq::cuda::runtime_options::window_input_views() &&
                input.is_contiguous() && ids.is_contiguous() && weights.is_contiguous() && weights.is_cuda() &&
                weights.scalar_type()==tb::kFloat32 && ids.device()==input.device() && weights.device()==input.device();
            if(cell->window_input_views)launch_body(*cell,execution,&input,&ids,&weights);
            else {
                cell->input.copy_(input);cell->ids.copy_(ids);cell->weights.copy_(weights);
                launch_body(*cell,execution);
            }
            cache_->moe_residency_->record_window_expert_fence(sources_[0]->layer_id_,stream_);
            return cell->output;
        }
        const auto* graph=cell->graphs->find(1,tokens);
        if(cell->pending) {
            if(cell->graph_replay) {
                if(!graph->wait_ms(60000))throw std::runtime_error("MFQ previous FFN graph did not complete");
            } else cell->eager_done->synchronize();
        }
        if(cell->pending)observe_gpu(*cell);
        cell->pending=false;std::memset(cell->flags.host,0,cell->flags.bytes);
        if(cell->graph_replay) {
            cell->input.copy_(input);cell->ids.copy_(ids);cell->weights.copy_(weights);
        } else {
            // Eager launches take the current tensors directly. Their storage
            // remains valid through the stream-ordered consumers; no graph
            // requires a fixed address between prefill chunks.
            cell->input=input;cell->ids=ids;cell->weights=weights;
        }
        std::string error;
        try {
            if(cell->graph_replay) {
                if(!graph->launch(stream_,error))throw std::runtime_error(error);
            } else {
                launch_body(*cell,execution);
                cell->eager_done->record(stream_);
            }
            cell->pending=true;
            serve(*cell);cache_->record_compute_use();
            std::vector<int32_t> routed(cell->host_ids.h<int32_t>(),cell->host_ids.h<int32_t>()+cell->entries);
            cache_->moe_residency_->after_layer(sources_[0]->layer_id_,routed,tokens);
        }
        catch(...) {failed_=true;publish(cell->flags.h<uint32_t>()+64);publish(cell->flags.h<uint32_t>()+80);for(int i=1;i<4;++i)publish(cell->flags.h<uint32_t>()+16*i);throw;}
        if(!cell->graph_replay) {
            auto output=std::move(cell->output);
            cell->input={};cell->ids={};cell->weights={};
            return output;
        }
        return cell->output;
    }
    friend MoeFfnForward make_moe_ffn_pipeline(const std::vector<std::shared_ptr<MfeWeight>>&,
        const std::vector<std::shared_ptr<MfeWeight>>&,MoeFfnShared,MoeFfnDispatchObserver,MoeFfnSharedWeights,MoeFfnPrefetch*);
};

MoeFfnForward make_moe_ffn_pipeline(const std::vector<std::shared_ptr<MfeWeight>>& gate_up,
        const std::vector<std::shared_ptr<MfeWeight>>& down,MoeFfnShared shared,MoeFfnDispatchObserver observer,
        MoeFfnSharedWeights shared_weights,MoeFfnPrefetch* prefetch) {
    if(gate_up.size()!=2 || down.size()!=1)return {};
    for(const auto& w:{gate_up[0],gate_up[1],down[0]})if(!w || !w->cached_source)return {};
    if(!gate_up[0]->cached_source->pipeline_enabled())return {};
    auto runtime=std::make_shared<MoeFfnPipeline>(std::array<std::shared_ptr<MoeCachedSource>,3>{
        gate_up[0]->cached_source,gate_up[1]->cached_source,down[0]->cached_source},std::move(shared),std::move(observer),std::move(shared_weights));
    if(prefetch)*prefetch=[runtime](const tb::Tensor& ids){return runtime->prediction(ids);};
    return [runtime](CudaExecutionContext& execution,const tb::Tensor& input,const tb::Tensor& ids,const tb::Tensor& weights){
        return runtime->forward(execution,input,ids,weights);
    };
}
