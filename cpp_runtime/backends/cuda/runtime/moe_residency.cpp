#include "moe_residency.h"
#include "runtime/decode_window.h"
#include "runtime/moe_pipeline.h"
#include "storage/moe_cached_source_internal.h"
#include "mfq/cpu_expert_pool.h"
#include "host_memory_copy.h"
#include "mfq/moe_residency_ranker.h"
#include <algorithm>
#include <array>
#include <exception>
#include <map>
#include <set>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdlib>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#include <fstream>
#endif

namespace tb=mfq_tensor_backend;
namespace {
class PhaseClock {
    std::atomic<std::uint64_t>& total_;
    std::atomic<std::uint64_t>* decode_;
    std::chrono::steady_clock::time_point start_=std::chrono::steady_clock::now();
public:
    PhaseClock(std::atomic<std::uint64_t>& total,std::atomic<std::uint64_t>* decode)
        :total_(total),decode_(decode) {}
    ~PhaseClock() {
        const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-start_).count();
        total_.fetch_add(elapsed,std::memory_order_relaxed);
        if(decode_)decode_->fetch_add(elapsed,std::memory_order_relaxed);
    }
};
MoeResidencyManager::MemorySample process_memory() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS process{};MEMORYSTATUSEX system{};system.dwLength=sizeof(system);
    if(!K32GetProcessMemoryInfo(GetCurrentProcess(),&process,sizeof(process)) || !GlobalMemoryStatusEx(&system))
        throw std::runtime_error("cannot measure physical memory for expert residency transaction");
    return {process.WorkingSetSize,system.ullAvailPhys};
#elif defined(__linux__)
    std::uint64_t virtual_pages=0,resident_pages=0;std::ifstream status("/proc/self/statm");
    struct sysinfo system{};
    if(!(status>>virtual_pages>>resident_pages) || sysinfo(&system)!=0)
        throw std::runtime_error("cannot measure physical memory for expert residency transaction");
    return {resident_pages*static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE)),
        std::uint64_t(system.freeram+system.bufferram)*system.mem_unit};
#else
    throw std::runtime_error("physical memory measurement unavailable on this CUDA host");
#endif
}
}
struct MoeResidencyManager::Impl {
    struct Move {
        int incoming_source=0,outgoing_source=0,cohort=0,incoming=0,outgoing=0,slot=-1;
        int incoming_bundle=0,incoming_projection=0;
        MoeHostExpertCache::Lease lease;
        mfq::MoeCacheSlotBook::Replacement replacement;
        std::vector<int64_t> offsets;
        std::size_t touched_fields=0;
        std::size_t backed_fields=0;
        MoeGpuArena* cached_arena=nullptr;
        int cached_slot=-1;
    };
    struct Map {int source=0,cohort=0;std::vector<int32_t> next;};
    MoeExpertCache* cache;
    Observer observer;
    MemoryProbe memory_probe;
    int first=0,last=0;
    int64_t rounds=0;
    int pending_tokens=0;
    std::vector<std::vector<float>> heat;
    const float heat_retention=mfq::cuda::runtime_options::residency_heat_retention();
    const int interval=mfq::cuda::runtime_options::residency_interval();
    const bool exchange_defaults=[this] {
        // The measured combination needs fixed GPU routing, registered RAM,
        // transfer-cache reuse and no background CPU calibration. Keep it a
        // coherent default: an explicit opt-out restores legacy defaults for
        // the other switches, while every explicit setting remains respected.
        if(!cache || !cache->complete_residency_ || !cache->config_.moe_residency_adapt ||
           !cache->config_.moe_residency_warm || !cache->config_.moe_residency_projection_heat ||
           !cache->config_.moe_direct_ram || !cache->pipeline_ram_registered_bytes_ ||
           !cache->pipeline_transfer_cache_ || !cache->config_.moe_ram_pcie_fraction ||
           cache->ram_pcie_fraction_!=1.0 || cache->pipeline_shared_cpu_cost_)return false;
        for(const auto* name:{"MFQ_MOE_RESIDENCY_DIRECT_EXCHANGE","MFQ_MOE_RESIDENCY_EXPERT_FENCE",
                "MFQ_MOE_RESIDENCY_CACHED_REFILL","MFQ_MOE_RESIDENCY_BATCHED",
                "MFQ_MOE_RESIDENCY_ASYNC_PUBLISH","MFQ_MOE_RESIDENCY_WINDOW_PREPARE"}) {
            const auto* value=std::getenv(name);if(value && value[0]!='1')return false;
        }
        const auto* promotion=std::getenv("MFQ_MOE_RESIDENCY_PROMOTE_TRANSFER");
        return !promotion || promotion[0]=='2';
    }();
    const bool promote_transfer=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_PROMOTE_TRANSFER");
        return flag?(flag[0]=='1' || flag[0]=='2'):exchange_defaults;
    }();
    const bool reuse_transfer=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_PROMOTE_TRANSFER");
        return flag?flag[0]=='2':exchange_defaults;
    }();
    std::vector<std::vector<float>> observations;
    std::unique_ptr<mfq::MoeResidencyRanker> ranking;
    struct TransferStage {std::string key;MoeGpuArena* arena=nullptr;};
    std::vector<std::vector<TransferStage>> transfer_stages;
    bool direct=false;
    bool batched=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_BATCHED");
        return flag?flag[0]!='0':exchange_defaults;
    }();
    bool async_publish=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_ASYNC_PUBLISH");
        return flag?flag[0]!='0':exchange_defaults;
    }();
    bool window_prepare=[] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_WINDOW_PREPARE");
        return !flag || flag[0]!='0';
    }();
    bool expert_fence=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_EXPERT_FENCE");
        return flag?flag[0]=='1':exchange_defaults;
    }();
    bool cached_refill=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_CACHED_REFILL");
        return flag?flag[0]=='1':exchange_defaults;
    }();
    bool direct_exchange=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_DIRECT_EXCHANGE");
        return flag?flag[0]=='1':exchange_defaults;
    }();
    std::uint64_t refill_dma_min=[] {
        const auto* value=std::getenv("MFQ_MOE_RESIDENCY_REFILL_DMA_MIN_BYTES");
        return value?std::uint64_t(std::max<long long>(0,std::atoll(value))):std::uint64_t(0);
    }();
    std::uint64_t refill_chunk_bytes=[] {
        const auto* value=std::getenv("MFQ_MOE_RESIDENCY_REFILL_CHUNK_BYTES");
        if(!value)return std::uint64_t(65536);
        const auto parsed=std::atoll(value);
        return parsed<=0?std::uint64_t(0):std::uint64_t(std::clamp<long long>(parsed,4096,1048576))&~std::uint64_t(15);
    }();
    bool parallel_host=[] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_PARALLEL_HOST");
        return flag && flag[0]!='0';
    }();
    std::atomic<std::uint64_t> observed_routes{0},scheduled_rounds{0},candidate_bundles{0};
    std::atomic<std::uint64_t> planned_bundles{0},committed_bundles{0},memory_rejections{0};
    std::atomic<std::uint64_t> host_pack_bytes{0},direct_upload_bytes{0},backup_peak_bytes{0},shared_backup_bytes{0};
    std::atomic<std::uint64_t> candidate_projections{0},planned_projections{0},committed_projections{0},projection_memory_rejections{0};
    std::atomic<std::uint64_t> prepare_ns{0},join_ns{0},dma_wait_ns{0},map_ns{0},host_exchange_ns{0},apply_ns{0};
    std::atomic<std::uint64_t> decode_prepare_ns{0},decode_join_ns{0},decode_dma_wait_ns{0},decode_map_ns{0},decode_host_exchange_ns{0},decode_apply_ns{0};
    std::atomic<std::uint64_t> backup_copies{0},upload_copies{0};
    std::atomic<std::uint64_t> parallel_host_bytes{0},parallel_host_batches{0};
    std::atomic<std::uint64_t> batch_device_bytes{0},batch_descriptor_copies{0};
    std::atomic<std::uint64_t> async_publish_ns{0},decode_async_publish_ns{0};
    std::atomic<std::uint64_t> window_prepares{0};
    std::atomic<std::uint64_t> cached_copy_bytes{0},cached_copy_fields{0};
    std::atomic<std::uint64_t> expert_fence_waits{0};
    std::atomic<std::uint64_t> cached_copy_batches{0};
    std::atomic<std::uint64_t> expert_fence_wait_ns{0},backup_submit_ns{0},refill_submit_ns{0};
    std::atomic<std::uint64_t> cached_copy_dma_bytes{0},cached_copy_dma_fields{0};
    std::atomic<std::uint64_t> direct_exchange_rounds{0},direct_exchange_bytes{0},direct_exchange_ns{0};
    std::atomic<std::uint64_t> cached_copy_descriptors{0},cached_copy_blocks{0};
    std::vector<Move> pending;
    std::size_t pending_complete_bundles=0;
    std::vector<Map> maps;
    tb::Tensor upload,backup;
    tb::Tensor device_work,copy_descriptors,device_descriptors;
    int work_tokens=0;
    cudaStream_t stream=nullptr;
    cudaEvent_t done=nullptr;
    cudaEvent_t window_done=nullptr;
    std::unique_ptr<mfq::cuda::HostBuffer> expert_done_host;
    uint32_t* expert_done_device=nullptr;
    cudaStream_t expert_compute_stream=nullptr;
    bool pending_window=false;
    bool pending_expert_fence=false;
    static constexpr std::size_t refill_capacity=65535;
    std::unique_ptr<mfq::cuda::HostBuffer> refill_descriptors;
    mfq::MoeCacheMappedCopyDescriptor* refill_device=nullptr;
    std::thread worker;
    std::mutex worker_mutex;
    std::condition_variable work_available,work_finished;
    std::atomic_bool work_pending{false};
    bool work_requested=false,work_ready=false,stopping=false,window_inflight=false;
    std::exception_ptr error;
    explicit Impl(MoeExpertCache* value,Observer observe,MemoryProbe probe)
        :cache(value),observer(std::move(observe)),memory_probe(probe?std::move(probe):process_memory) {}
    void observe(const char* phase,std::size_t index) {if(observer)observer(phase,index);}
    std::size_t complete_pending_bundles() const {
        std::map<std::pair<int,int>,unsigned> masks;
        for(const auto& move:pending)masks[{move.incoming_bundle,move.incoming}]|=1u<<move.incoming_projection;
        return std::count_if(masks.begin(),masks.end(),[](const auto& item){return item.second==7;});
    }
    struct CachedRefill {
        Impl& s;
        std::size_t count=0;
        std::uint64_t largest=0;
        explicit CachedRefill(Impl& owner):s(owner) {}
        void copy(uint8_t* destination,const uint8_t* source,std::uint64_t bytes) {
            const auto chunk=s.refill_chunk_bytes?s.refill_chunk_bytes:bytes;
            const auto pieces=1+(bytes-1)/chunk;
            if(s.cached_refill && (!s.refill_dma_min || bytes<s.refill_dma_min) && pieces<=refill_capacity-count) {
                auto* descriptors=static_cast<mfq::MoeCacheMappedCopyDescriptor*>(s.refill_descriptors->data());
                for(std::uint64_t begin=0;begin<bytes;begin+=chunk) {
                    const auto size=std::min(chunk,bytes-begin);
                    descriptors[count++]={reinterpret_cast<uint64_t>(destination+begin),reinterpret_cast<uint64_t>(source+begin),size};
                    largest=std::max(largest,size);
                }
            } else {
                MFQ_CUDA_CHECK(cudaMemcpyAsync(destination,source,std::size_t(bytes),cudaMemcpyDeviceToDevice,s.stream));
                s.cached_copy_dma_bytes+=bytes;++s.cached_copy_dma_fields;
            }
            s.cached_copy_bytes+=bytes;++s.cached_copy_fields;
        }
        bool submit() {
            if(!count)return false;
            // Chunked descriptors receive one block each. Work grows with
            // actual bytes, avoiding maximum-size grids for tiny fields.
            const int blocks=s.refill_chunk_bytes?1:static_cast<int>(std::min<std::uint64_t>(128,(largest+4095)/4096));
            std::atomic_thread_fence(std::memory_order_seq_cst);
            mfq::moe_cache_mapped_gather_cuda(s.refill_device,int(count),blocks,s.stream);
            ++s.cached_copy_batches;s.cached_copy_descriptors+=count;s.cached_copy_blocks+=count*blocks;
            return true;
        }
    };
};

MoeResidencyManager::MoeResidencyManager(MoeExpertCache* cache,Observer observer,MemoryProbe memory_probe)
    :impl_(std::make_unique<Impl>(cache,std::move(observer),std::move(memory_probe))) {
    if(!cache || !cache->complete_residency_ || cache->pipeline_bundles_.empty())
        throw std::invalid_argument("adaptive tier requires resident expert bundles");
    auto& s=*impl_;s.first=std::numeric_limits<int>::max();s.last=std::numeric_limits<int>::min();
    s.transfer_stages.resize(cache->sources_.size());
    for(const auto& bundle:cache->pipeline_bundles_) {
        const auto& source=cache->sources_.at(bundle[0]);s.heat.emplace_back(std::size_t(source->n_experts()),0.0f);
        s.observations.emplace_back(std::size_t(source->n_experts()),0.0f);
        for(int id:bundle)if(cache->sources_.at(id)->layer_id_!=source->layer_id_ ||
                cache->sources_.at(id)->n_experts()!=source->n_experts())
            throw std::invalid_argument("adaptive expert bundle geometry differs");
        s.first=std::min(s.first,source->layer_id_);s.last=std::max(s.last,source->layer_id_);
        for(int p=0;p<3;++p) {
            const auto& projection=*cache->sources_[bundle[p]];
            auto& stages=s.transfer_stages[bundle[p]];
            for(const auto& cohort:projection.cohorts_)
                stages.push_back({std::to_string(p)+":"+cohort.arena->signature,nullptr});
        }
    }
    using Expert=mfq::MoeResidencyRanker::Expert;
    std::vector<std::vector<Expert>> layout;
    if(cache->config_.moe_residency_warm && cache->config_.moe_residency_projection_heat) {
        std::map<MoeGpuArena*,std::vector<Expert>> groups;
        for(std::size_t b=0;b<cache->pipeline_bundles_.size();++b) {
            const auto& bundle=cache->pipeline_bundles_[b];
            for(int e=0;e<cache->sources_[bundle[0]]->n_experts();++e)for(int p=0;p<3;++p) {
                const auto& source=*cache->sources_[bundle[p]];
                groups[source.cohorts_[source.expert_to_cohort_[e]].arena].push_back({int(b),e,p});
            }
        }
        for(auto& item:groups)layout.push_back(std::move(item.second));
    } else {
        std::map<std::array<std::uintptr_t,3>,std::vector<Expert>> groups;
        for(std::size_t b=0;b<cache->pipeline_bundles_.size();++b) {
            const auto& bundle=cache->pipeline_bundles_[b];
            for(int e=0;e<cache->sources_[bundle[0]]->n_experts();++e) {
                std::array<std::uintptr_t,3> signature;
                for(int p=0;p<3;++p) {
                    const auto& source=*cache->sources_[bundle[p]];
                    signature[p]=reinterpret_cast<std::uintptr_t>(source.cohorts_[source.expert_to_cohort_[e]].arena);
                }
                groups[signature].push_back({int(b),e,-1});
            }
        }
        for(auto& item:groups)layout.push_back(std::move(item.second));
    }
    s.ranking=std::make_unique<mfq::MoeResidencyRanker>(std::move(layout));
    if(s.cached_refill) {
        // Allocate before any consumer graph runs. Only descriptors are mapped;
        // weights stay in their original GPU fields. This avoids a per-round
        // allocation or descriptor upload delaying the early D2H backups.
        s.refill_descriptors=std::make_unique<mfq::cuda::HostBuffer>(
            Impl::refill_capacity*sizeof(mfq::MoeCacheMappedCopyDescriptor),true);
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&s.refill_device),s.refill_descriptors->data(),0));
    }
    MFQ_CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking));
    const auto event=cudaEventCreateWithFlags(&s.done,cudaEventDisableTiming);
    if(event!=cudaSuccess){cudaStreamDestroy(s.stream);s.stream=nullptr;MFQ_CUDA_CHECK(event);}
    const auto window_event=cudaEventCreateWithFlags(&s.window_done,cudaEventDisableTiming);
    if(window_event!=cudaSuccess){cudaEventDestroy(s.done);s.done=nullptr;cudaStreamDestroy(s.stream);s.stream=nullptr;MFQ_CUDA_CHECK(window_event);}
}
MoeResidencyManager::~MoeResidencyManager() {
    if(impl_->worker.joinable()) {
        {std::lock_guard<std::mutex> lock(impl_->worker_mutex);impl_->stopping=true;}
        impl_->work_available.notify_one();impl_->work_finished.notify_all();impl_->worker.join();
    }
    try {if(!impl_->pending.empty())rollback();}catch(...) {}
    // Batch storage is allocated on the adaptation stream. Its async frees
    // must be issued while that stream is still alive.
    impl_->device_work={};impl_->device_descriptors={};
    impl_->copy_descriptors={};impl_->upload={};impl_->backup={};
    if(impl_->stream)(void)cudaStreamSynchronize(impl_->stream);
    if(impl_->done)cudaEventDestroy(impl_->done);
    if(impl_->window_done)cudaEventDestroy(impl_->window_done);
    if(impl_->stream)cudaStreamDestroy(impl_->stream);
}
void MoeResidencyManager::before_layer(int layer) {
    auto& s=*impl_;if(layer!=s.first)return;
    apply_pending();
    if(s.expert_done_host) {
        *static_cast<volatile uint32_t*>(s.expert_done_host->data())=0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
}
bool MoeResidencyManager::prepare_during_window() const noexcept {
    return impl_->window_prepare && !impl_->cache->pipeline_prefill_layer_layout_;
}
void MoeResidencyManager::record_window_expert_fence(int layer,void* raw_stream) {
    auto& s=*impl_;
    if(!s.expert_fence || !s.window_prepare || s.cache->pipeline_prefill_layer_layout_ || layer!=s.last)return;
    const auto stream=static_cast<cudaStream_t>(raw_stream);
    cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
    MFQ_CUDA_CHECK(cudaStreamIsCapturing(stream,&status));
    if(!s.expert_done_host) {
        if(status!=cudaStreamCaptureStatusNone)
            throw std::logic_error("adaptive expert fence was not warmed before graph capture");
        s.expert_done_host=std::make_unique<mfq::cuda::HostBuffer>(128,true);
        std::memset(s.expert_done_host->data(),0,128);
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&s.expert_done_device),s.expert_done_host->data(),0));
    }
    // The final expert output is complete here; the LM head and residual work
    // never access expert weights or transfer slots. Only GPU backup/upload may
    // start early. RAM ownership and mapping publication retain finish_window.
    // Match the mapped publication protocol used by Strata/MFQ routing. A
    // replayed external CUDA event did not release side-stream copies before
    // a blocked graph tail in the local WDDM regression test.
    s.expert_compute_stream=stream;
    mfq::cuda::signal_mapped_flag(s.expert_done_device,stream);
}
void MoeResidencyManager::finish_window(int layer) {
    auto& s=*impl_;if(layer!=s.last)return;
    if(s.window_prepare)s.cache->record_compute_use();
    {std::lock_guard<std::mutex> lock(s.worker_mutex);s.window_inflight=false;}
    s.work_finished.notify_all();
}
void MoeResidencyManager::after_layer(int layer,const std::vector<int32_t>& ids,int tokens,bool window_inflight) {
    auto& s=*impl_;
    // Resident exchanges overwrite the initial cold complement used by the
    // field-major layer images. Keep their ownership fixed in this opt-in mode.
    if(s.cache->pipeline_prefill_layer_layout_)return;
    for(std::size_t b=0;b<s.cache->pipeline_bundles_.size();++b)
        if(s.cache->sources_[s.cache->pipeline_bundles_[b][0]]->layer_id_==layer)
            for(int expert:ids){s.observations[b].at(expert)+=1.0f;++s.observed_routes;}
    if(!s.cache->config_.moe_residency_adapt)return;
    if(layer!=s.last || s.work_pending.load(std::memory_order_acquire))return;
    if(!s.cache->config_.moe_residency_warm && (tokens>8 || (++s.rounds%4)!=0))return;
    // Strata adjusts its cache every few completed windows. Keep collecting
    // all routes between rounds; prompt batches still prepare immediately.
    if(s.cache->config_.moe_residency_warm && tokens==1 && (++s.rounds%s.interval)!=0)return;
    if(!s.pending.empty())throw std::logic_error("adaptive tier has an unpublished transaction");
    for(std::size_t b=0;b<s.heat.size();++b)for(std::size_t e=0;e<s.heat[b].size();++e) {
        s.heat[b][e]+=s.observations[b][e];s.observations[b][e]=0;
    }
    ++s.scheduled_rounds;
    s.pending_tokens=tokens;
    s.error=nullptr;
    // During replay the entire graph is already queued on the compute stream.
    // Planning can proceed now; resident writes must follow its completion.
    s.pending_window=window_inflight;
    s.pending_expert_fence=window_inflight && s.expert_fence && bool(s.expert_done_host);
    if(window_inflight){MFQ_CUDA_CHECK(cudaEventRecord(s.window_done,mfq_get_current_cuda_stream().stream()));++s.window_prepares;}
    if(!s.worker.joinable())s.worker=std::thread([this] {
        auto& state=*impl_;
        std::unique_lock<std::mutex> lock(state.worker_mutex);
        while(true) {
            state.work_available.wait(lock,[&]{return state.work_requested || state.stopping;});
            if(!state.work_requested)break;
            state.work_requested=false;lock.unlock();
            try {
                prepare();
                if(state.async_publish) {
                    std::unique_lock<std::mutex> finish_lock(state.worker_mutex);
                    state.work_finished.wait(finish_lock,[&]{return !state.window_inflight || state.stopping;});
                    const bool publish=!state.stopping;finish_lock.unlock();
                    const PhaseClock publish_clock(state.async_publish_ns,
                        state.pending_tokens<=8?&state.decode_async_publish_ns:nullptr);
                    if(publish)publish_pending();
                }
            }
            catch(...) {
                auto failure=std::current_exception();
                try {rollback();}catch(...) {failure=std::current_exception();}
                state.error=failure;
            }
            lock.lock();state.work_ready=true;state.work_finished.notify_one();
        }
    });
    s.work_pending.store(true,std::memory_order_release);
    {std::lock_guard<std::mutex> lock(s.worker_mutex);s.window_inflight=window_inflight;s.work_requested=true;}
    s.work_available.notify_one();
}
void MoeResidencyManager::prepare() {
    auto& s=*impl_;auto& cache=*s.cache;
    if(cache.pipeline_prefill_layer_layout_)return;
    const PhaseClock clock(s.prepare_ns,s.pending_tokens<=8?&s.decode_prepare_ns:nullptr);
    const MfqCudaGuard guard(cache.sources_.front()->cohorts_.front().active.expert_local.device());
    using Expert=mfq::MoeResidencyRanker::Expert;
    if(cache.pipeline_transfer_cache_)for(auto& stages:s.transfer_stages)for(auto& stage:stages)
        if(!stage.arena) {
            const auto found=cache.pipeline_stages_.find(stage.key);
            if(found!=cache.pipeline_stages_.end())stage.arena=found->second.get();
        }
    auto& pairs=s.ranking->rank([&](const Expert& e){return s.heat[e.bundle][e.expert];},
        [&](const Expert& e) {
            const auto& bundle=cache.pipeline_bundles_[e.bundle];
            if(e.projection>=0) {
                const auto& source=*cache.sources_[bundle[e.projection]];
                if(source.quant_source_->gpu_resident(e.expert))return 1;
                // A transfer-cache hit is temporary residency. Let a measured
                // hot projection compete for permanent space when requested;
                // the normal lease, inflight and publication guards still
                // protect the RAM/GPU exchange.
                if(cache.pipeline_transfer_cache_ && !s.promote_transfer) {
                    const int c=source.expert_to_cohort_[e.expert];
                    const auto* arena=s.transfer_stages[bundle[e.projection]][c].arena;
                    if(arena && arena->book && arena->book->slot_for({source.id_,c,e.expert})>=0)return -1;
                }
                return 0;
            }
            bool hot=true,cold=true;
            for(int p=0;p<3;++p) {
                const bool gpu=cache.sources_[bundle[p]]->quant_source_->gpu_resident(e.expert);
                hot&=gpu;cold&=!gpu;
            }
            return hot?1:cold?0:-1;
        });
    std::map<std::pair<int,int>,unsigned> candidates;
    for(const auto& pair:pairs) {
        candidates[{pair.incoming.bundle,pair.incoming.expert}]|=pair.incoming.projection<0?7:1u<<pair.incoming.projection;
        s.candidate_projections+=pair.incoming.projection<0?3:1;
    }
    s.candidate_bundles+=std::count_if(candidates.begin(),candidates.end(),[](const auto& item){return item.second==7;});
    if(!cache.config_.moe_residency_warm && pairs.size()>32)pairs.resize(32);
    s.direct=cache.config_.moe_direct_ram && cache.pipeline_ram_registered_bytes_>0;
    const bool shared_backup=cache.config_.moe_residency_warm && s.direct && cache.pipeline_host_stage_.defined();
    // The next layer waits for publication before using the transfer workspace.
    // Reuse its pinned storage between windows instead of allocating a second
    // model-sized backup alongside graph preparation.
    if(shared_backup) {
        s.backup=cache.pipeline_host_stage_;
        s.shared_backup_bytes=s.backup.numel();
    }
    std::uint64_t stage_limit=std::numeric_limits<std::uint64_t>::max();
    if(cache.config_.moe_residency_warm) {
        const auto memory=s.memory_probe();auto available=memory.available_bytes;
        if(cache.config_.moe_host_physical_bytes)
            available=std::min(available,memory.resident_bytes>=cache.config_.moe_host_physical_bytes
                ? std::uint64_t(0):cache.config_.moe_host_physical_bytes-memory.resident_bytes);
        const auto reusable=std::uint64_t(s.backup.defined()?s.backup.numel():0)+
            std::uint64_t(s.upload.defined()?s.upload.numel():0);
        stage_limit=(available+reusable)/(s.direct?1:2);
        if(shared_backup)stage_limit=std::min(stage_limit,std::uint64_t(s.backup.numel()));
    }
    s.pending.reserve(pairs.size()*3);int64_t bytes=0;
    std::set<std::pair<int,int>> memory_rejections;
    for(const auto& pair:pairs) {
        const int count=pair.incoming.projection<0?3:1;
        std::vector<Impl::Move> bundle(count);bool valid=true;
        for(int i=0;i<count;++i) {
            const int incoming_projection=count==3?i:pair.incoming.projection;
            const int outgoing_projection=count==3?i:pair.outgoing.projection;
            auto& move=bundle[i];move.incoming_source=cache.pipeline_bundles_[pair.incoming.bundle][incoming_projection];
            move.outgoing_source=cache.pipeline_bundles_[pair.outgoing.bundle][outgoing_projection];
            move.incoming_bundle=pair.incoming.bundle;move.incoming_projection=incoming_projection;
            move.incoming=pair.incoming.expert;move.outgoing=pair.outgoing.expert;
            auto& incoming=*cache.sources_[move.incoming_source];auto& outgoing=*cache.sources_[move.outgoing_source];
            move.cohort=incoming.expert_to_cohort_[move.incoming];auto& cohort=incoming.cohorts_[move.cohort];
            const auto old_key=outgoing.quant_source_->host_key(move.outgoing);move.slot=cohort.arena->book->slot_for(old_key);
            if(move.slot<0 || cohort.arena->book->inflight(move.slot) ||
                    !cache.host_experts_->contains(incoming.quant_source_->host_key(move.incoming))){valid=false;break;}
            move.lease=incoming.quant_source_->acquire_expert(move.incoming);
            if(move.lease.use_count()!=2){valid=false;break;}
            if(s.reuse_transfer && s.direct && cache.pipeline_transfer_cache_) {
                auto* arena=s.transfer_stages[move.incoming_source][move.cohort].arena;
                const int slot=arena && arena->book ? arena->book->slot_for(
                    incoming.quant_source_->host_key(move.incoming)) : -1;
                if(slot>=0){move.cached_arena=arena;move.cached_slot=slot;}
            }
            move.replacement=cohort.arena->book->prepare_replace(move.slot,old_key,incoming.quant_source_->host_key(move.incoming));
        }
        if(!valid)continue;
        int64_t next_bytes=bytes;
        for(auto& move:bundle) {
            auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
            for(auto count:cohort.bytes_per_expert) {
                next_bytes=(next_bytes+15)&~int64_t(15);move.offsets.push_back(next_bytes);
                if(count>INT64_MAX-next_bytes)throw std::overflow_error("adaptive staging size overflow");next_bytes+=count;
            }
        }
        if(std::uint64_t(next_bytes)>stage_limit){memory_rejections.emplace(pair.incoming.bundle,pair.incoming.expert);s.projection_memory_rejections+=count;continue;}
        bytes=next_bytes;for(auto& move:bundle)s.pending.push_back(std::move(move));
    }
    s.memory_rejections+=memory_rejections.size();
    for(auto& heat:s.heat)for(float& value:heat)value*=s.heat_retention;
    if(s.pending.empty())return;
    const auto options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true);
    if(cache.config_.moe_residency_warm &&
        (!s.backup.defined() || s.backup.numel()<bytes || (!s.direct && (!s.upload.defined() || s.upload.numel()<bytes)))) {
        s.backup={};s.upload={};
    }
    if(!s.backup.defined() || s.backup.numel()<bytes)s.backup=tb::empty({bytes},options);
    if(!s.direct && (!s.upload.defined() || s.upload.numel()<bytes))s.upload=tb::empty({bytes},options);
    s.backup_peak_bytes=std::max(s.backup_peak_bytes.load(),std::uint64_t(s.backup.numel()));
    s.pending_complete_bundles=s.complete_pending_bundles();
    s.planned_bundles+=s.pending_complete_bundles;s.planned_projections+=s.pending.size();
    auto update_map=[&](int source,int cohort,int expert,int slot) {
        auto found=std::find_if(s.maps.begin(),s.maps.end(),[&](const auto& item){return item.source==source && item.cohort==cohort;});
        if(found==s.maps.end()){s.maps.push_back({source,cohort,cache.sources_[source]->cohorts_[cohort].host_map});found=std::prev(s.maps.end());}
        found->next[expert]=slot;
    };
    for(auto& move:s.pending) {
        auto& source=*cache.sources_[move.incoming_source];auto& cohort=source.cohorts_[move.cohort];
        auto& outgoing=*cache.sources_[move.outgoing_source];
        update_map(move.incoming_source,move.cohort,move.incoming,move.slot);
        update_map(move.outgoing_source,outgoing.expert_to_cohort_[move.outgoing],move.outgoing,-1);
        cohort.arena->book->mark_inflight(move.slot);
        if(!s.direct) {
            const auto fields=moe_cache_fields(move.lease->weights);
            for(std::size_t f=0;f<fields.size();++f) {
                std::memcpy(s.upload.data_ptr<uint8_t>()+move.offsets[f],fields[f].data_ptr(),std::size_t(cohort.bytes_per_expert[f]));
                s.host_pack_bytes+=cohort.bytes_per_expert[f];
            }
        }
    }
    if(s.pending_window) {
        if(s.pending_expert_fence) {
            const PhaseClock wait_clock(s.expert_fence_wait_ns,nullptr);
            mfq::cuda::wait_route_publication(static_cast<const uint32_t*>(s.expert_done_host->data()),s.expert_compute_stream,60000);
            ++s.expert_fence_waits;
        } else MFQ_CUDA_CHECK(cudaStreamWaitEvent(s.stream,s.window_done,0));
    }
    else if(cache.compute_done_recorded_)MFQ_CUDA_CHECK(cudaStreamWaitEvent(s.stream,cache.compute_done_,0));
    // Only fixed GPU routing without background CPU calibration uses this
    // path. The host-cache lock spans the exchange, so window completion must
    // never require a new host-cache lookup. Existing readers reject it.
    if(s.direct_exchange && s.direct && s.reuse_transfer &&
            (!s.pending_window || s.pending_expert_fence) &&
            cache.config_.moe_ram_pcie_fraction && cache.ram_pcie_fraction_==1.0 &&
            !cache.pipeline_shared_cpu_cost_ &&
            std::all_of(cache.pipeline_dispatch_replay_.begin(),cache.pipeline_dispatch_replay_.end(),
                [](const auto& record){return record.cpu.empty();}) &&
            std::all_of(s.pending.begin(),s.pending.end(),[](const auto& move){return move.cached_arena!=nullptr;})) {
        prepare_direct_exchange();return;
    }
    // The packed path starts with a GPU gather. On the tested WDDM runtime,
    // that side-stream work cannot pass a running graph's tail; even a fully
    // preallocated gather reproduces the stall. Queue D2H backups on the copy
    // engine for an early expert fence, preserving the existing GPU-cache reuse
    // and RAM/publication transaction. Normal windows retain packed batching.
    if(s.batched && !s.pending_expert_fence && !s.cached_refill) {
        prepare_batched(bytes);
        MFQ_CUDA_CHECK(cudaEventRecord(s.done,s.stream));
        if(s.observer) {MFQ_CUDA_CHECK(cudaEventSynchronize(s.done));s.observe("copies_complete",0);}
        return;
    }
    {
    const PhaseClock backup_clock(s.backup_submit_ns,nullptr);
    for(const auto& move:s.pending) {
        auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        for(std::size_t f=0;f<move.offsets.size();++f) {
            const auto count=cohort.bytes_per_expert[f];if(!count)continue;
            auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
            MFQ_CUDA_CHECK(cudaMemcpyAsync(s.backup.data_ptr<uint8_t>()+move.offsets[f],gpu,std::size_t(count),cudaMemcpyDeviceToHost,s.stream));
            ++s.backup_copies;
        }
    }
    }
    {
    const PhaseClock refill_clock(s.refill_submit_ns,nullptr);
    Impl::CachedRefill refill(s);
    for(std::size_t i=0;i<s.pending.size();++i) {
        auto& move=s.pending[i];auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        const auto incoming=moe_cache_fields(move.lease->weights);
        for(std::size_t f=0;f<move.offsets.size();++f) {
            const auto count=cohort.bytes_per_expert[f];if(!count)continue;
            auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
            move.touched_fields=f+1;
            if(move.cached_arena) {
                const auto* cached=static_cast<const uint8_t*>(move.cached_arena->fields[f].data_ptr())+int64_t(move.cached_slot)*count;
                // The window event fences the final transfer scatter. The
                // next window joins this transaction before reusing its slots.
                refill.copy(gpu,cached,std::uint64_t(count));
            } else {
                const auto* host=s.direct?static_cast<const uint8_t*>(incoming[f].data_ptr()):s.upload.data_ptr<uint8_t>()+move.offsets[f];
                MFQ_CUDA_CHECK(cudaMemcpyAsync(gpu,host,std::size_t(count),cudaMemcpyHostToDevice,s.stream));
                ++s.upload_copies;
                if(s.direct)s.direct_upload_bytes+=count;
            }
        }
        s.observe("upload",i);
    }
    // All backups precede this kernel; publication still waits for done.
    refill.submit();
    }
    MFQ_CUDA_CHECK(cudaEventRecord(s.done,s.stream));
    if(s.observer) {MFQ_CUDA_CHECK(cudaEventSynchronize(s.done));s.observe("copies_complete",0);}
}
void MoeResidencyManager::prepare_direct_exchange() {
    auto& s=*impl_;auto& cache=*s.cache;
    const PhaseClock direct_clock(s.direct_exchange_ns,nullptr);
    // This callback runs before the cache lock, allowing an arriving reader to
    // retain its lease. exchange_batch must then reject without changing bytes.
    s.observe("before_direct_exchange",0);
    std::vector<MoeHostExpertCache::Exchange> changes;changes.reserve(s.pending.size());
    for(std::size_t i=0;i<s.pending.size();++i) {
        auto& move=s.pending[i];
        auto& incoming=*cache.sources_[move.incoming_source];
        auto& outgoing=*cache.sources_[move.outgoing_source];
        changes.push_back({incoming.quant_source_->host_key(move.incoming),
            outgoing.quant_source_->host_key(move.outgoing),&move.lease,
            [&,i](const MixedMoePool& reused) {
                const auto& item=s.pending[i];auto& old=*cache.sources_[item.outgoing_source];
                const auto& cohort=old.cohorts_[old.expert_to_cohort_[item.outgoing]];
                s.observe("host",i);auto result=*cohort.cpu;result.local_experts=1;result.expert_local={};
                return moe_replace_quant_fields(std::move(result),moe_cache_fields(reused));
            },[&,i](const MixedMoePool& reused) {
                auto& item=s.pending[i];
                const auto& cohort=cache.sources_[item.incoming_source]->cohorts_[item.cohort];
                const auto fields=moe_cache_fields(reused);
                MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
                // RAM now contains the victim. Restore its GPU bytes before
                // restoring the incoming RAM payload from its retained cache.
                for(std::size_t f=0;f<item.touched_fields;++f) {
                    const auto count=cohort.bytes_per_expert[f];if(!count)continue;
                    auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(item.slot)*count;
                    MFQ_CUDA_CHECK(cudaMemcpyAsync(gpu,fields[f].data_ptr(),std::size_t(count),cudaMemcpyHostToDevice,s.stream));
                }
                for(std::size_t f=0;f<item.backed_fields;++f) {
                    const auto count=cohort.bytes_per_expert[f];if(!count)continue;
                    const auto* cached=static_cast<const uint8_t*>(item.cached_arena->fields[f].data_ptr())+int64_t(item.cached_slot)*count;
                    MFQ_CUDA_CHECK(cudaMemcpyAsync(fields[f].data_ptr(),cached,std::size_t(count),cudaMemcpyDeviceToHost,s.stream));
                }
                MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
                item.touched_fields=0;item.backed_fields=0;
            }});
    }
    const bool committed=cache.host_experts_->exchange_batch(changes,[&] {
        {
            const PhaseClock wait_clock(s.dma_wait_ns,s.pending_tokens<=8?&s.decode_dma_wait_ns:nullptr);
            MFQ_CUDA_CHECK(cudaEventSynchronize(s.done));
            s.observe("copies_complete",0);
        }
        // Final GPU expert consumers have finished before prepare() enters
        // this path. Keep registry and GPU map publication at window finish.
        {
            std::unique_lock<std::mutex> lock(s.worker_mutex);
            s.work_finished.wait(lock,[&]{return !s.window_inflight || s.stopping;});
            if(s.stopping)throw std::runtime_error("adaptive direct exchange cancelled before window finish");
        }
        {
            const PhaseClock map_clock(s.map_ns,s.pending_tokens<=8?&s.decode_map_ns:nullptr);
            for(const auto& move:s.pending)
                if(!cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->replacement_valid(move.replacement))
                    throw std::runtime_error("adaptive GPU slot changed before direct publication");
            for(std::size_t i=0;i<s.maps.size();++i) {
                const auto& map=s.maps[i];auto& cohort=cache.sources_[map.source]->cohorts_[map.cohort];
                MFQ_CUDA_CHECK(cudaMemcpyAsync(cohort.active.expert_local.data_ptr<int32_t>(),map.next.data(),
                    map.next.size()*sizeof(int32_t),cudaMemcpyHostToDevice,s.stream));s.observe("map",i);
            }
            MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
        }
        s.observe("publish",0);
        try {
            for(auto& move:s.pending)
                if(!cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->commit_replace(move.replacement))
                    throw std::runtime_error("adaptive direct replacement lost its generation");
        } catch(...) {
            for(auto& move:s.pending)
                cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->rollback_replace(move.replacement);
            throw;
        }
        for(auto& map:s.maps) {
            auto& cohort=cache.sources_[map.source]->cohorts_[map.cohort];
            cohort.host_map.swap(map.next);cohort.map_dirty=false;
        }
        for(auto& move:s.pending) {
            cache.sources_[move.outgoing_source]->quant_source_->publish_gpu_resident(move.outgoing,false);
            cache.sources_[move.incoming_source]->quant_source_->publish_gpu_resident(move.incoming,true);
        }
    },[&] {
        {
            const PhaseClock backup_clock(s.backup_submit_ns,nullptr);
            for(auto& move:s.pending) {
                const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
                const auto fields=moe_cache_fields(move.lease->weights);
                for(std::size_t f=0;f<fields.size();++f) {
                    const auto count=cohort.bytes_per_expert[f];if(!count)continue;
                    const auto* gpu=static_cast<const uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
                    move.backed_fields=f+1;
                    MFQ_CUDA_CHECK(cudaMemcpyAsync(fields[f].data_ptr(),gpu,std::size_t(count),cudaMemcpyDeviceToHost,s.stream));
                    ++s.backup_copies;s.observe("direct_backup",f);
                }
            }
        }
        {
            const PhaseClock refill_clock(s.refill_submit_ns,nullptr);
            Impl::CachedRefill refill(s);
            for(std::size_t i=0;i<s.pending.size();++i) {
                auto& move=s.pending[i];const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
                for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                    const auto count=cohort.bytes_per_expert[f];if(!count)continue;
                    auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
                    const auto* cached=static_cast<const uint8_t*>(move.cached_arena->fields[f].data_ptr())+int64_t(move.cached_slot)*count;
                    move.touched_fields=f+1;
                    // All victims have already been backed up into final RAM.
                    // Mark queued fields before launch so either submission or
                    // publication failures restore the complete transaction.
                    refill.copy(gpu,cached,std::uint64_t(count));
                    s.observe("direct_refill",f);
                }
                s.observe("upload",i);
            }
            if(refill.submit())s.observe("direct_refill_batch",0);
        }
        MFQ_CUDA_CHECK(cudaEventRecord(s.done,s.stream));
    });
    if(!committed){rollback();return;}
    s.committed_bundles+=s.pending_complete_bundles;s.committed_projections+=s.pending.size();
    ++s.direct_exchange_rounds;
    for(const auto& move:s.pending) {
        ++cache.stats_.evictions;
        const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        cache.stats_.gpu_demote_bytes+=cohort.arena->slot_bytes;
        for(auto count:cohort.bytes_per_expert)s.direct_exchange_bytes+=count;
    }
    s.pending.clear();s.maps.clear();
}

void MoeResidencyManager::prepare_batched(std::int64_t bytes) {
    auto& s=*impl_;auto& cache=*s.cache;
    // cudaMallocAsync must be ordered with the copy kernels using its storage,
    // independently of a model stream that may still be executing its tail.
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    const MfqCudaStreamGuard stream_guard(MfqCudaStream(mfq_current_cuda_device(),s.stream));
#else
    const MfqCudaStreamGuard stream_guard(at::cuda::getStreamFromExternal(s.stream,mfq_current_cuda_device()));
#endif
    struct Field {const uint8_t* host;uint8_t* gpu;int64_t bytes,backup_offset,upload_offset;const uint8_t* cached;};
    struct Interval {const uint8_t* host;int64_t bytes,offset;};
    std::vector<Field> fields;
    int64_t largest=0;
    for(const auto& move:s.pending) {
        const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        const auto incoming=moe_cache_fields(move.lease->weights);
        for(std::size_t f=0;f<move.offsets.size();++f) {
            const auto count=cohort.bytes_per_expert[f];if(!count)continue;
            auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
            const auto* cached=move.cached_arena ? static_cast<const uint8_t*>(move.cached_arena->fields[f].data_ptr())+
                int64_t(move.cached_slot)*count : nullptr;
            fields.push_back({s.direct?static_cast<const uint8_t*>(incoming[f].data_ptr()):nullptr,
                gpu,count,move.offsets[f],move.offsets[f],cached});
            largest=std::max(largest,count);
        }
    }
    if(fields.empty() || fields.size()>65535)
        throw std::overflow_error("adaptive copy descriptors exceed CUDA grid geometry");
    std::vector<Interval> intervals;
    int64_t upload_bytes=bytes;
    if(s.direct) {
        std::sort(fields.begin(),fields.end(),[](const Field& a,const Field& b) {
            return reinterpret_cast<uintptr_t>(a.host)<reinterpret_cast<uintptr_t>(b.host);
        });
        upload_bytes=0;
        for(std::size_t first=0;first<fields.size();) {
            if(fields[first].cached){++first;continue;}
            const auto address=reinterpret_cast<uintptr_t>(fields[first].host);
            auto end=address+fields[first].bytes;auto last=first+1;
            while(last<fields.size()) {
                if(fields[last].cached)break;
                const auto next=reinterpret_cast<uintptr_t>(fields[last].host);
                if(next<end || next-end>15)break;
                end=next+fields[last].bytes;++last;
            }
            upload_bytes=(upload_bytes+15)&~int64_t(15);
            const auto count=static_cast<int64_t>(end-address);
            if(count>INT64_MAX-upload_bytes)throw std::overflow_error("adaptive upload interval overflow");
            for(auto i=first;i<last;++i)fields[i].upload_offset=upload_bytes+
                static_cast<int64_t>(reinterpret_cast<uintptr_t>(fields[i].host)-address);
            intervals.push_back({fields[first].host,count,upload_bytes});
            upload_bytes+=count;first=last;
        }
    }
    const auto gpu_options=s.cache->sources_.front()->cohorts_.front().active.expert_local.options().dtype(tb::kUInt8);
    const auto host_options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true);
    const int64_t work_bytes=std::max(bytes,upload_bytes);
    const int64_t descriptor_bytes=int64_t(fields.size())*sizeof(mfq::MoeCacheMappedCopyDescriptor);
    // These copies are outside captured graphs. Release a larger prefill
    // workspace at a row-count transition; steady decode keeps reusing it.
    if(s.work_tokens!=s.pending_tokens) {
        if(s.device_work.defined() && s.device_work.numel()>work_bytes)s.device_work={};
        if(s.copy_descriptors.defined() && s.copy_descriptors.numel()>2*descriptor_bytes) {
            s.copy_descriptors={};s.device_descriptors={};
        }
    }
    s.work_tokens=s.pending_tokens;
    s.observe("workspace_begin",0);
    if(!s.device_work.defined() || s.device_work.numel()<work_bytes)s.device_work=tb::empty({work_bytes},gpu_options);
    s.observe("workspace_device",0);
    if(!s.copy_descriptors.defined() || s.copy_descriptors.numel()<2*descriptor_bytes) {
        s.copy_descriptors=tb::empty({2*descriptor_bytes},host_options);
        s.observe("workspace_host_descriptors",0);
        s.device_descriptors=tb::empty({2*descriptor_bytes},gpu_options);
    }
    s.observe("workspace_ready",0);
    s.batch_device_bytes=s.device_work.numel()+s.device_descriptors.numel();
    auto* work=s.device_work.data_ptr<uint8_t>();
    auto* host=reinterpret_cast<mfq::MoeCacheMappedCopyDescriptor*>(s.copy_descriptors.data_ptr<uint8_t>());
    auto* device=reinterpret_cast<mfq::MoeCacheMappedCopyDescriptor*>(s.device_descriptors.data_ptr<uint8_t>());
    for(std::size_t f=0;f<fields.size();++f) {
        const auto& field=fields[f];
        host[f]={reinterpret_cast<uint64_t>(work+field.backup_offset),reinterpret_cast<uint64_t>(field.gpu),uint64_t(field.bytes)};
        host[fields.size()+f]={reinterpret_cast<uint64_t>(field.gpu),
            reinterpret_cast<uint64_t>(field.cached?field.cached:work+field.upload_offset),uint64_t(field.bytes)};
        if(field.cached){s.cached_copy_bytes+=field.bytes;++s.cached_copy_fields;}
    }
    MFQ_CUDA_CHECK(cudaMemcpyAsync(device,host,std::size_t(2*descriptor_bytes),cudaMemcpyHostToDevice,s.stream));
    ++s.batch_descriptor_copies;
    // One device workspace serves both directions. The stream completes the
    // outgoing backup before reusing its bytes or changing any resident field.
    const int blocks=static_cast<int>(std::min<int64_t>(128,(largest+4095)/4096));
    mfq::moe_cache_mapped_gather_cuda(device,int(fields.size()),blocks,s.stream);
    MFQ_CUDA_CHECK(cudaMemcpyAsync(s.backup.data_ptr(),work,std::size_t(bytes),cudaMemcpyDeviceToHost,s.stream));
    ++s.backup_copies;
    s.observe("backup",0);
    if(s.direct) {
        for(const auto& interval:intervals) {
            MFQ_CUDA_CHECK(cudaMemcpyAsync(work+interval.offset,interval.host,std::size_t(interval.bytes),cudaMemcpyHostToDevice,s.stream));
            ++s.upload_copies;s.direct_upload_bytes+=interval.bytes;
        }
    } else {
        MFQ_CUDA_CHECK(cudaMemcpyAsync(work,s.upload.data_ptr(),std::size_t(bytes),cudaMemcpyHostToDevice,s.stream));
        ++s.upload_copies;
    }
    for(auto& move:s.pending)move.touched_fields=move.offsets.size();
    mfq::moe_cache_mapped_gather_cuda(device+fields.size(),int(fields.size()),blocks,s.stream);
    for(std::size_t i=0;i<s.pending.size();++i)s.observe("upload",i);
}
void MoeResidencyManager::rollback() {
    auto& s=*impl_;auto& cache=*s.cache;
    const MfqCudaGuard guard(cache.sources_.front()->cohorts_.front().active.expert_local.device());
    MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
    for(auto& move:s.pending) {
        auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        for(std::size_t f=0;f<move.touched_fields;++f) {
            const auto count=cohort.bytes_per_expert[f];auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
            MFQ_CUDA_CHECK(cudaMemcpyAsync(gpu,s.backup.data_ptr<uint8_t>()+move.offsets[f],std::size_t(count),cudaMemcpyHostToDevice,s.stream));
        }
    }
    for(const auto& map:s.maps) {
        auto& cohort=cache.sources_[map.source]->cohorts_[map.cohort];
        MFQ_CUDA_CHECK(cudaMemcpyAsync(cohort.active.expert_local.data_ptr<int32_t>(),cohort.host_map.data(),
            cohort.host_map.size()*sizeof(int32_t),cudaMemcpyHostToDevice,s.stream));
    }
    MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
    for(auto& move:s.pending)cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->rollback_replace(move.replacement);
    s.pending.clear();s.maps.clear();
}
void MoeResidencyManager::apply_pending() {
    auto& s=*impl_;
    if(!s.work_pending.load(std::memory_order_acquire) && s.pending.empty() && !s.error)return;
    const bool decode=s.pending_tokens<=8;
    const PhaseClock apply_clock(s.apply_ns,decode?&s.decode_apply_ns:nullptr);
    {
        const PhaseClock join_clock(s.join_ns,decode?&s.decode_join_ns:nullptr);
        if(s.work_pending.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(s.worker_mutex);
            s.work_finished.wait(lock,[&]{return s.work_ready;});s.work_ready=false;
            s.work_pending.store(false,std::memory_order_release);
        }
    }
    if(s.error){auto failure=s.error;s.error=nullptr;std::rethrow_exception(failure);}
    publish_pending();
}
void MoeResidencyManager::publish_pending() {
    auto& s=*impl_;
    if(s.pending.empty())return;
    const bool decode=s.pending_tokens<=8;
    auto& cache=*s.cache;const MfqCudaGuard guard(cache.sources_.front()->cohorts_.front().active.expert_local.device());
    try {
        {
        const PhaseClock dma_clock(s.dma_wait_ns,decode?&s.decode_dma_wait_ns:nullptr);
        MFQ_CUDA_CHECK(cudaEventSynchronize(s.done));
        if(!cache.pipeline_dma_leases_.empty()) {
            MFQ_CUDA_CHECK(cudaEventSynchronize(cache.transfer_ready_));
            cache.pipeline_dma_leases_.clear();
        }
        }
        {
        const PhaseClock map_clock(s.map_ns,decode?&s.decode_map_ns:nullptr);
        for(const auto& move:s.pending)if(!cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->replacement_valid(move.replacement))
            throw std::runtime_error("adaptive GPU slot changed before publication");
        for(std::size_t i=0;i<s.maps.size();++i) {
            const auto& map=s.maps[i];auto& cohort=cache.sources_[map.source]->cohorts_[map.cohort];
            MFQ_CUDA_CHECK(cudaMemcpyAsync(cohort.active.expert_local.data_ptr<int32_t>(),map.next.data(),
                map.next.size()*sizeof(int32_t),cudaMemcpyHostToDevice,s.stream));s.observe("map",i);
        }
        MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
        }
        const PhaseClock host_clock(s.host_exchange_ns,decode?&s.decode_host_exchange_ns:nullptr);
        std::vector<MoeHostExpertCache::Exchange> changes;changes.reserve(s.pending.size());
        struct HostCopy {uint8_t* destination;const uint8_t* source;int64_t begin,end;};
        std::vector<HostCopy> copies;int64_t copy_bytes=0;
        for(std::size_t i=0;i<s.pending.size();++i) {
            auto& move=s.pending[i];auto& incoming=*cache.sources_[move.incoming_source];auto& outgoing=*cache.sources_[move.outgoing_source];
            {
                const auto fields=moe_cache_fields(move.lease->weights);
                const auto& cohort=incoming.cohorts_[move.cohort];
                for(std::size_t f=0;f<fields.size();++f) {
                    const auto count=cohort.bytes_per_expert[f];if(!count)continue;
                    copies.push_back({static_cast<uint8_t*>(fields[f].data_ptr()),
                        s.backup.data_ptr<uint8_t>()+move.offsets[f],copy_bytes,copy_bytes+count});
                    copy_bytes+=count;
                }
            }
            changes.push_back({incoming.quant_source_->host_key(move.incoming),outgoing.quant_source_->host_key(move.outgoing),&move.lease,
                [&,i](const MixedMoePool& reused) {
                    const auto& item=s.pending[i];auto& old=*cache.sources_[item.outgoing_source];
                    const auto& cohort=old.cohorts_[old.expert_to_cohort_[item.outgoing]];auto fields=moe_cache_fields(reused);
                    s.observe("host",i);auto result=*cohort.cpu;result.local_experts=1;result.expert_local={};
                    return moe_replace_quant_fields(std::move(result),fields);
                },[&,i](const MixedMoePool& reused) {
                    const auto& item=s.pending[i];const auto& cohort=cache.sources_[item.incoming_source]->cohorts_[item.cohort];auto fields=moe_cache_fields(reused);
                    for(std::size_t f=0;f<fields.size();++f) {
                        const auto count=cohort.bytes_per_expert[f];
                        if(s.direct) {
                            const auto* gpu=static_cast<const uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(item.slot)*count;
                            MFQ_CUDA_CHECK(cudaMemcpyAsync(fields[f].data_ptr(),gpu,std::size_t(count),cudaMemcpyDeviceToHost,s.stream));
                        } else std::memcpy(fields[f].data_ptr(),s.upload.data_ptr<uint8_t>()+item.offsets[f],std::size_t(count));
                    }
                    if(s.direct)MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
                }});
        }
        std::function<void()> prepare_host;
        if(s.parallel_host)prepare_host=[&] {
            mfq::cpu::expert_pool().rows(copy_bytes,[&](int64_t begin,int64_t end) {
                auto copy=std::upper_bound(copies.begin(),copies.end(),begin,
                    [](int64_t position,const HostCopy& item){return position<item.end;});
                while(begin<end) {
                    const auto count=std::min(end,copy->end)-begin;
                    std::memcpy(copy->destination+(begin-copy->begin),copy->source+(begin-copy->begin),std::size_t(count));
                    begin+=count;++copy;
                }
            });
            s.parallel_host_bytes+=copy_bytes;++s.parallel_host_batches;
        };
        if(!s.parallel_host)prepare_host=[&] {
            mfq::cpu::HostMemoryCopyBatch batch;
            for(const auto& copy:copies)
                batch.copy(copy.destination,copy.source,std::size_t(copy.end-copy.begin));
        };
        const bool committed=cache.host_experts_->exchange_batch(changes,[&] {
            s.observe("publish",0);
            try {
                for(auto& move:s.pending)if(!cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->commit_replace(move.replacement))
                    throw std::runtime_error("adaptive GPU replacement lost its generation");
            } catch(...) {
                for(auto& move:s.pending)cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->rollback_replace(move.replacement);
                throw;
            }
            for(auto& map:s.maps){auto& cohort=cache.sources_[map.source]->cohorts_[map.cohort];cohort.host_map.swap(map.next);cohort.map_dirty=false;}
            for(auto& move:s.pending) {
                cache.sources_[move.outgoing_source]->quant_source_->publish_gpu_resident(move.outgoing,false);
                cache.sources_[move.incoming_source]->quant_source_->publish_gpu_resident(move.incoming,true);
            }
        },prepare_host);
        if(!committed){rollback();return;}
        s.committed_bundles+=s.pending_complete_bundles;s.committed_projections+=s.pending.size();
        for(const auto& move:s.pending){++cache.stats_.evictions;cache.stats_.gpu_demote_bytes+=cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->slot_bytes;}
        s.pending.clear();s.maps.clear();
    } catch(...) {const auto failure=std::current_exception();rollback();std::rethrow_exception(failure);}
}
MoeResidencyManager::Stats MoeResidencyManager::stats() const noexcept {
    const auto& s=*impl_;return {s.observed_routes.load(),s.scheduled_rounds.load(),s.candidate_bundles.load(),
        s.planned_bundles.load(),s.committed_bundles.load(),s.memory_rejections.load(),
        s.host_pack_bytes.load(),s.direct_upload_bytes.load(),s.backup_peak_bytes.load(),s.shared_backup_bytes.load(),
        s.candidate_projections.load(),s.planned_projections.load(),s.committed_projections.load(),s.projection_memory_rejections.load(),
        s.prepare_ns.load(),s.join_ns.load(),s.dma_wait_ns.load(),s.map_ns.load(),s.host_exchange_ns.load(),s.apply_ns.load(),
        s.decode_prepare_ns.load(),s.decode_join_ns.load(),s.decode_dma_wait_ns.load(),s.decode_map_ns.load(),s.decode_host_exchange_ns.load(),s.decode_apply_ns.load(),
        s.backup_copies.load(),s.upload_copies.load(),s.parallel_host,s.parallel_host_bytes.load(),s.parallel_host_batches.load(),
        s.batched,s.batch_device_bytes.load(),s.batch_descriptor_copies.load(),
        s.async_publish,s.async_publish_ns.load(),s.decode_async_publish_ns.load(),s.window_prepares.load(),
        s.cached_copy_bytes.load(),s.cached_copy_fields.load(),s.expert_fence_waits.load(),s.cached_copy_batches.load(),
        s.expert_fence_wait_ns.load(),s.backup_submit_ns.load(),s.refill_submit_ns.load(),
        s.cached_copy_dma_bytes.load(),s.cached_copy_dma_fields.load(),
        s.direct_exchange_rounds.load(),s.direct_exchange_bytes.load(),s.direct_exchange_ns.load(),
        s.cached_copy_descriptors.load(),s.cached_copy_blocks.load(),s.exchange_defaults};
}
