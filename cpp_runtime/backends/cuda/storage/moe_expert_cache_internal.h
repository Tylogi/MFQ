#include "../runtime/execution_options.h"
#pragma once

#include "moe_cache_types_internal.h"
#include "mfq_cuda_context.h"
#include "runtime/moe_residency.h"
#include "runtime/dma_copy_batch.h"
#include "runtime/kv_expert_loan.h"
#include "mfq/host_parallel.h"
#include "mfq/moe_cpu_cost_model.h"
#include "mfq/moe_cpu_calibration.h"
#include <array>
#include <cstring>
#include <fstream>
#include <future>
#include <condition_variable>

struct MoeFfnPrefillHostWorkspace;
struct CudaProfiler;
class MoePinnedExpertArena;
struct MoeSsdPrefillReader;
class MoeExpertCache {
public:
    struct PrefillCopySubmission {
        int layer=-1,slot=-1;
        int copies=0,direct_copies=0;
        int64_t direct_bytes=0,total_bytes=0,gate_up_bytes=0,down_bytes=0;
        bool phased=false;
        std::shared_ptr<CudaProfiler> profile;
    };
    void record_prefill_copy(const PrefillCopySubmission&);
    PrefillCopySubmission finish_prefill_copy(int slot);
    void finish_prefill_copies();
    std::shared_ptr<mfq::cuda::KvExpertLoan> kv_expert_loan_;
    std::vector<int> set_prefill_low_priority_sources(std::vector<int>);
    std::vector<int> prioritize_prefill_layers(int first,int last);
    bool prefill_layer_active(int layer) const {
        return !std::binary_search(prefill_low_priority_layers_.begin(),prefill_low_priority_layers_.end(),layer);
    }
    void prepare_direct_transfer_cache();
    void release_direct_transfer_cache();
    std::size_t begin_prefill_loan(std::size_t workspace_bytes);
    void suspend_prefill_loan();
    void end_prefill_loan();
    bool prefill_loan_active() const {return bool(prefill_loan_);}
    std::size_t prefill_transfer_bytes(bool predictor=false) const;
    MoePrefillWorkspaceBudget prefill_workspace_budget(bool allow_borrow) const;
    std::optional<std::size_t> available_host_ram() const;
    std::size_t reclaim_optional_host(std::size_t allocation_bytes);
    void reserve_workspace(std::size_t bytes) {if(reserve_workspace_)reserve_workspace_(bytes);}
    void set_workspace_reservation(std::function<void(std::size_t)> reserve,
                                   std::function<void(std::size_t)> reserve_ram={}) {
        reserve_workspace_=std::move(reserve);reserve_ram_workspace_=std::move(reserve_ram);
    }
    bool layer_prefill_ready() const { return pipeline_prefill_layer_layout_ && pipeline_ram_registered_bytes_>0; }
    std::uint64_t streamed_bytes() const {return stats_.ram_pcie_bytes;}
    MoeExpertCache(
            int64_t budget_bytes,
            const CudaExecutionConfig& config)
        : budget_bytes_(budget_bytes),config_(config) {
        if (budget_bytes_ <= 0) {
            throw std::invalid_argument(
                "MoE GPU cache budget must be positive");
        }
        MFQ_CUDA_CHECK(cudaStreamCreateWithFlags(
            &weight_stream_, cudaStreamNonBlocking));
        MFQ_CUDA_CHECK(cudaStreamCreateWithFlags(
            &route_stream_, cudaStreamNonBlocking));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &compute_done_, cudaEventDisableTiming));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &transfer_ready_, cudaEventDisableTiming));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &route_input_ready_, cudaEventDisableTiming));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &route_done_, cudaEventDisableTiming));
        stages_.resize(4);
        host_experts_=std::make_shared<MoeHostExpertCache>(config.moe_host_cache_bytes);
        for (auto & stage : stages_) {
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &stage.done, cudaEventDisableTiming));
        }
        mapped_gather_enabled_ = config.moe_mapped_gather;
        mapped_copy_blocks_ = config.moe_mapped_copy_blocks;
        range_read_pool_ =
            std::make_unique<mfq::cuda::MfeMxfp4ReadPool>(
                config.moe_ssd_io_workers);
        range_overlap_enabled_ = config.moe_ssd_overlap;
        const auto diagnostics=mfq::cuda::runtime_options::moe_diagnostics();
        pipeline_cpu_profile_enabled_=diagnostics.cpu_rows;
        pipeline_cpu_profile_fail_alloc_=diagnostics.cpu_rows_fail_alloc;
        pipeline_dma_profile_enabled_=diagnostics.dma;
        pipeline_dma_profile_fail_alloc_=diagnostics.dma_fail_alloc;
        pipeline_shared_cpu_cost_=diagnostics.shared_cpu_cost;
        pipeline_dispatch_record_path_=diagnostics.dispatch_record;
        pipeline_miss_record_path_=diagnostics.miss_record;
        pipeline_dispatch_capture_required_=!pipeline_miss_record_path_.empty();
        if(diagnostics.dispatch_replay) {
            std::ifstream input(*diagnostics.dispatch_replay);if(!input)throw std::runtime_error("cannot open required MFQ dispatch replay");
            std::string line;
            while(std::getline(input,line)) {
                DispatchRecord row;int64_t routes=0,cpu=0;std::istringstream fields(line);
                if(!(fields>>row.layer>>row.tokens>>routes>>cpu) || routes<=0 || cpu<0 || cpu>routes)
                    throw std::runtime_error("invalid MFQ dispatch replay geometry");
                row.ids.resize(routes);row.cpu.resize(cpu);
                for(auto& id:row.ids)if(!(fields>>id))throw std::runtime_error("truncated MFQ dispatch replay routes");
                for(auto& id:row.cpu)if(!(fields>>id))throw std::runtime_error("truncated MFQ dispatch replay CPU experts");
                std::string extra;if(fields>>extra)throw std::runtime_error("unexpected MFQ dispatch replay data");
                pipeline_dispatch_replay_.push_back(std::move(row));
            }
            if(!input.eof() || pipeline_dispatch_replay_.empty())throw std::runtime_error("empty or unreadable MFQ dispatch replay");
        }
    }

    ~MoeExpertCache() {
        ssd_prediction_reader_.reset();
        ssd_prediction_reads_.reset();ssd_previous_prediction_.reset();
        ssd_prefill_reader_.reset();
        try{finish_prefill_copies();}catch(const std::exception& error) {
            std::cerr<<"runtime_prefill_copy destructor_error="<<error.what()<<'\n';
        }
        if(prefill_loan_)try{end_prefill_loan();}catch(const std::exception& e) {
            std::cerr<<"runtime_prefill_loan destructor_error="<<e.what()<<'\n';
        }
        pipeline_cpu_calibration_.wait();
        if(pipeline_prefill_layer_buffer_.defined() && weight_stream_)
            (void)cudaStreamSynchronize(weight_stream_);
        if(pipeline_prefill_pool_)
            (void)cudaMemPoolSetAttribute(pipeline_prefill_pool_,cudaMemPoolAttrReleaseThreshold,
                &pipeline_prefill_pool_previous_threshold_);
        moe_residency_.reset();
        try {release_direct_transfer_cache();}catch(const std::exception& error) {
            std::cerr<<"runtime_direct_expert_cache destructor_error="<<error.what()<<'\n';
        }
        if(pipeline_pool_context_) {
            (void)cudaStreamSynchronize(pipeline_pool_stream_);
        }
        // Drain and destroy copy graphs while their stream and graph buffers
        // are still alive. Every layer shares these two phase caches.
        for(auto& batch:pipeline_dma_batches_)batch.reset();
        if(pipeline_pool_context_)
            pipeline_pool_context_->end_graph_pool(pipeline_pool_stream_);
        if ((!registered_host_fields_.empty() || pinned_expert_arena_ ||
                pipeline_prefill_host_ring_.defined() || pipeline_prefill_metadata_host_.defined()) &&
                weight_stream_ != nullptr) {
            (void)cudaStreamSynchronize(weight_stream_);
        }
        for (auto it = registered_host_fields_.rbegin();
             it != registered_host_fields_.rend();
             ++it) {
            if (it->owned) {
                (void)cudaHostUnregister(it->host);
            }
        }
        for (auto & stage : stages_) {
            if (stage.done != nullptr) cudaEventDestroy(stage.done);
        }
        for(auto event:pipeline_prefill_layer_ready_)if(event)cudaEventDestroy(event);
        for(auto event:pipeline_prefill_layer_gate_up_ready_)if(event)cudaEventDestroy(event);
        for(auto event:pipeline_prefill_layer_read_done_)if(event)cudaEventDestroy(event);
        if (compute_done_ != nullptr) cudaEventDestroy(compute_done_);
        if (transfer_ready_ != nullptr) {
            cudaEventDestroy(transfer_ready_);
        }
        if (route_input_ready_ != nullptr) {
            cudaEventDestroy(route_input_ready_);
        }
        if (route_done_ != nullptr) {
            cudaEventDestroy(route_done_);
        }
        if (route_stream_ != nullptr) cudaStreamDestroy(route_stream_);
        if (weight_stream_ != nullptr) cudaStreamDestroy(weight_stream_);
    }

    std::shared_ptr<MoeCachedSource> register_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> cpu,
        int minimum_slots,
        int layer_id,
        std::string projection_role);

    std::shared_ptr<MoeCachedSource> register_range_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> metadata,
        std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> store,
        int minimum_slots,
        int layer_id,
        std::string projection_role);

    std::shared_ptr<MoeCachedSource> register_quant_range_source(
        const std::string& name,std::shared_ptr<MoeQuantRangeSource> source,
        int minimum_slots,int layer_id,std::string projection_role);

    MoeGpuArena * register_cohort(
            const MixedMoePool & pool,
            int out_per_expert,
            int neuron_len,
            int minimum_slots) {
        if (finalized_) {
            throw std::runtime_error(
                "cannot register a MoE source after cache finalization");
        }
        validate_nepq_expert_boundaries(
            pool, out_per_expert, neuron_len);
        return register_cohort_layout(
            pool,
            out_per_expert,
            neuron_len,
            minimum_slots,
            pool.local_experts,
            moe_cache_field_layouts(pool));
    }

    MoeGpuArena * register_cohort_layout(
            const MixedMoePool & pool,
            int out_per_expert,
            int neuron_len,
            int minimum_slots,
            int registered_experts,
            std::vector<MoeCacheFieldLayout> layouts) {
        if (finalized_) {
            throw std::runtime_error(
                "cannot register a MoE source after cache finalization");
        }
        if (registered_experts <= 0 || layouts.empty()) {
            throw std::runtime_error("invalid MoE cache cohort layout");
        }
        const std::string signature =
            moe_cache_signature(pool, out_per_expert, neuron_len, layouts);
        auto found = arenas_.find(signature);
        if (found == arenas_.end()) {
            auto arena = std::make_unique<MoeGpuArena>();
            arena->signature = signature;
            arena->minimum_slots =
                std::min(minimum_slots, registered_experts);
            arena->registered_experts = registered_experts;
            for (const auto & layout : layouts) {
                if (layout.slot_shape.empty() || layout.elements < 0 ||
                        layout.element_size <= 0) {
                    throw std::runtime_error(
                        "invalid MoE cache field layout");
                }
                arena->slot_bytes +=
                    layout.elements * layout.element_size;
            }
            arena->layouts = std::move(layouts);
            MoeGpuArena * result = arena.get();
            arenas_.emplace(signature, std::move(arena));
            return result;
        }
        MoeGpuArena * arena = found->second.get();
        arena->minimum_slots = std::max(
            arena->minimum_slots,
            std::min(minimum_slots, registered_experts));
        arena->registered_experts += registered_experts;
        if (arena->layouts.size() != layouts.size()) {
            throw std::runtime_error(
                "MoE cache signature merged incompatible field counts");
        }
        for (size_t index = 0; index < layouts.size(); ++index) {
            const auto & left = arena->layouts[index];
            const auto & right = layouts[index];
            if (left.scalar_type != right.scalar_type ||
                    left.slot_shape != right.slot_shape ||
                    left.elements != right.elements ||
                    left.element_size != right.element_size) {
                throw std::runtime_error(
                    "MoE cache signature merged incompatible field layouts");
            }
        }
        return arena;
    }

    void finalize(std::size_t kv_growth_reserve = 0, std::size_t kv_budget_charge = 0);
    void resize(std::size_t bytes);
    std::size_t minimum_bytes() const {
        std::size_t result=0;
        for(const auto& [signature,arena]:arenas_)result+=std::size_t(arena->minimum_slots)*arena->slot_bytes;
        return result;
    }

    void set_profile(mfq::MoeCacheProfile profile) {
        if (finalized_ || !sources_.empty()) {
            throw std::runtime_error(
                "MoE cache profile must be set before source registration");
        }
        profile_ = std::move(profile);
    }

    bool finalized() const noexcept {
        return finalized_;
    }

    bool has_sources() const noexcept {
        return !sources_.empty();
    }

    int64_t budget_bytes() const noexcept {
        return budget_bytes_;
    }

    int64_t allocated_bytes() const noexcept {
        return allocated_bytes_;
    }

    const uint8_t * register_mapped_field(
            const mfq_tensor_backend::Tensor & field) {
        if (!mapped_gather_enabled_ || !field.defined() ||
                field.numel() == 0) {
            return nullptr;
        }
        if (!field.is_cpu() || !field.is_contiguous()) {
            throw std::runtime_error(
                "mapped MoE cache fields must be contiguous CPU tensors");
        }
        auto * host = field.data_ptr();
        const auto existing = mapped_host_lookup_.find(host);
        if (existing != mapped_host_lookup_.end()) {
            return existing->second;
        }
        const int64_t bytes = tensor_nbytes(field);
        if (bytes <= 0 ||
                static_cast<uint64_t>(bytes) >
                    std::numeric_limits<size_t>::max()) {
            throw std::runtime_error(
                "mapped MoE cache field has an invalid byte count");
        }
        cudaError_t status = cudaHostRegister(
            host,
            static_cast<size_t>(bytes),
            cudaHostRegisterMapped);
        bool owned = true;
        if (status == cudaErrorHostMemoryAlreadyRegistered) {
            (void)cudaGetLastError();
            owned = false;
        } else {
            MFQ_CUDA_CHECK(status);
        }
        void * device = nullptr;
        status = cudaHostGetDevicePointer(&device, host, 0);
        if (status != cudaSuccess) {
            if (owned) (void)cudaHostUnregister(host);
            MFQ_CUDA_CHECK(status);
        }
        auto * mapped = reinterpret_cast<const uint8_t *>(device);
        mapped_host_lookup_.emplace(host, mapped);
        registered_host_fields_.push_back({host, bytes, owned});
        mapped_registered_bytes_ += bytes;
        return mapped;
    }

    const MoeCacheStats & stats() const noexcept {
        return stats_;
    }

    bool prepare(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch);

    bool prepare_bundle(
        const std::vector<MoeCachedSource *> & sources,
        const std::vector<int32_t> & experts);

    bool prepare_bundle_deferred(
        const std::vector<MoeCachedSource *> & ready_sources,
        MoeCachedSource & deferred_source,
        const std::vector<int32_t> & experts);

    void prewarm();
    void preload_complete_residency();
    void initialize_ssd_residency();
    void bind_host_device_cache_hints();
    void initialize_ssd_readers();
    struct SsdPlacementStats {std::size_t experts=0,bytes=0,batches=0,peak=0;};
    void prepare_ssd_placement(MoeCachedSource& source,const std::vector<int32_t>& experts,SsdPlacementStats& stats);
    using SsdExpertRequest=std::pair<MoeQuantRangeSource*,int>;
    struct SsdReadPriority {
        std::mutex mutex;
        std::condition_variable idle;
        // Count unfinished urgent fields, including queued work. Completed
        // fields release their capacity while the remaining batch is consumed.
        std::size_t pending=0,waiting=0;
        struct Demand {
            SsdReadPriority& priority;
            std::size_t remaining;
            explicit Demand(SsdReadPriority& value,std::size_t jobs=1):priority(value),remaining(jobs) {
                std::lock_guard lock(priority.mutex);priority.pending+=remaining;
            }
            Demand(const Demand&)=delete;
            Demand& operator=(const Demand&)=delete;
            void complete() {
                {
                    std::lock_guard lock(priority.mutex);
                    if(!remaining)throw std::logic_error("SSD demand completion exceeded its registered fields");
                    --remaining;--priority.pending;
                }
                priority.idle.notify_all();
            }
            ~Demand() {
                {std::lock_guard lock(priority.mutex);priority.pending-=remaining;}
                priority.idle.notify_all();
            }
        };
        bool wait_for_capacity(std::size_t urgent_allowance,const std::function<bool()>& canceled={},
                bool* overlapped=nullptr) {
            const auto ready=[&]{return pending<=urgent_allowance || (canceled && canceled());};
            std::unique_lock lock(mutex);
            const bool deferred=!ready();
            if(deferred){++waiting;idle.wait(lock,ready);--waiting;}
            if(overlapped)*overlapped=pending>0 && pending<=urgent_allowance;
            return deferred;
        }
        void notify_canceled() {
            // Pair notification with the wait mutex even though cancellation
            // itself is atomic, so it cannot fall between the predicate and wait.
            std::lock_guard lock(mutex);idle.notify_all();
        }
    };
    struct SsdExpertReads {
        std::vector<SsdExpertRequest> requests;
        std::vector<std::size_t> request_bytes;
        static constexpr std::size_t early_down_budget=16u<<20;
        std::vector<MoeHostExpertCache::Lease> leases;
        std::vector<std::size_t> missing;
        std::future<void> ready;
        std::optional<SsdReadPriority::Demand> demand;
        std::mutex mutex;
        std::condition_variable changed;
        std::size_t gate_up_requests=0,gate_up_remaining=0,down_remaining=0;
        std::vector<std::size_t> gate_up_ready,down_ready;
        bool publish_progress=false;
        std::exception_ptr failed;
        std::unique_ptr<std::atomic<uint8_t>[]> prediction_states;
        void initialize_predictions() {
            prediction_states=std::make_unique<std::atomic<uint8_t>[]>(missing.size());
        }
        bool prediction_canceled(std::size_t index) const {
            return prediction_states[index].load(std::memory_order_acquire)==2;
        }
        bool start_prediction(std::size_t index) {
            uint8_t queued=0;
            return prediction_states[index].compare_exchange_strong(queued,1,std::memory_order_acq_rel);
        }
        std::pair<std::size_t,std::size_t> cancel_unused_predictions(
                const std::vector<SsdExpertRequest>& demand,
                const std::array<MoeQuantRangeSource*,3>& current_sources) {
            if(!prediction_states || std::none_of(requests.begin(),requests.end(),[&](const auto& request) {
                return std::find(current_sources.begin(),current_sources.end(),request.first)!=current_sources.end();
            }))return {};
            std::size_t jobs=0,bytes=0;
            for(std::size_t i=0;i<missing.size();++i) {
                const auto index=missing[i];
                if(std::find(demand.begin(),demand.end(),requests[index])!=demand.end())continue;
                uint8_t queued=0;
                if(prediction_states[i].compare_exchange_strong(queued,2,std::memory_order_acq_rel)) {
                    ++jobs;bytes+=request_bytes[index];
                }
            }
            return {jobs,bytes};
        }
        void prioritize_prediction_experts(const std::vector<int>& experts) {
            const auto rank=[&](std::size_t index) {
                return std::find(experts.begin(),experts.end(),requests[index].second)-experts.begin();
            };
            std::stable_sort(missing.begin(),missing.end(),[&](auto a,auto b){return rank(a)<rank(b);});
        }
        bool prioritize_missing() {
            // Start the largest reads first to reduce the final-worker tail.
            // GU keeps priority over Down; ready and RAM-hit indices stay fixed.
            const auto earlier=[&](std::size_t a,std::size_t b) {
                const bool gu_a=a<gate_up_requests,gu_b=b<gate_up_requests;
                if(gu_a!=gu_b)return gu_a;
                if(request_bytes[a]!=request_bytes[b])return request_bytes[a]>request_bytes[b];
                return a<b;
            };
            if(std::is_sorted(missing.begin(),missing.end(),earlier))return false;
            std::sort(missing.begin(),missing.end(),earlier);return true;
        }
        ~SsdExpertReads(){if(ready.valid())ready.wait();}
        void wait(){if(ready.valid())ready.get();}
        void publish_read(std::size_t index) {
            {
                std::lock_guard lock(mutex);
                const bool down=index>=gate_up_requests;
                (down?down_ready:gate_up_ready).push_back(index);
                --(down?down_remaining:gate_up_remaining);
            }
            changed.notify_one();
        }
        bool next_projection_batch(std::vector<std::size_t>& batch,bool down,
                const std::function<bool()>& progress={},bool include_ready_down=false,
                const std::function<bool()>& down_idle={}) {
            std::unique_lock lock(mutex);
            auto& queue=down?down_ready:gate_up_ready;
            const auto& remaining=down?down_remaining:gate_up_remaining;
            const bool allow_down=include_ready_down && !down;
            const auto ready=[&]{return !queue.empty() || !remaining || failed ||
                (allow_down && !down_ready.empty());};
            for(;;) {
                if(failed)std::rethrow_exception(failed);
                if(!queue.empty() || !remaining)break;
                bool pending=false;
                if(allow_down && !down_ready.empty()) {
                    bool idle=true;
                    if(down_idle) {
                        lock.unlock();idle=down_idle();lock.lock();
                    }
                    if(failed)std::rethrow_exception(failed);
                    // GU may arrive while checking the transfer event.
                    if(!queue.empty() || !remaining)continue;
                    if(idle && !down_ready.empty()) {
                        // One idle-stream quantum may contain several small
                        // projections. Bound bytes ahead of a later GU arrival;
                        // a single oversized projection must still make progress.
                        std::size_t count=0,bytes=0;
                        do {
                            const auto size=request_bytes.at(down_ready[count]);
                            if(count && size>early_down_budget-bytes)break;
                            bytes+=size;++count;
                        }while(count<down_ready.size() && bytes<early_down_budget);
                        batch.assign(down_ready.begin(),down_ready.begin()+count);
                        down_ready.erase(down_ready.begin(),down_ready.begin()+count);
                        return true;
                    }
                    pending=!idle;
                }
                // A pending DMA completion has no condition-variable signal.
                // Poll it without holding the file-read publication mutex.
                // Keep the producer responsive to either completion instead
                // of sleeping for an OS timer tick between GPU event polls.
                if(progress) {
                    lock.unlock();const bool active=progress();lock.lock();
                    pending=active || pending;
                }
                if(!pending)changed.wait(lock,ready);
            }
            batch.clear();
            batch.swap(queue);
            return !batch.empty();
        }
    };
    std::unique_ptr<SsdExpertReads> prepare_ssd_expert_reads(std::vector<SsdExpertRequest> requests,bool background=false,bool prediction=false);
    void execute_ssd_expert_reads(SsdExpertReads& reads,bool background=false);
    void execute_ssd_expert_reads(SsdExpertReads& reads,mfq::HostParallelPool& pool,int workers,bool touch=true,
        SsdReadPriority::Demand* demand=nullptr);
    void queue_ssd_prediction(std::vector<SsdExpertRequest> requests);
    void observe_ssd_prediction_demand(const std::vector<SsdExpertRequest>& requests,
        const std::array<MoeQuantRangeSource*,3>& current_sources);
    void observe_ssd_read_wait(std::uint64_t ns) {
        if(!pipeline_reuse_learning_)return;
        ssd_prediction_wait_ns_=(ssd_prediction_wait_ns_*15+ns)/16;
        // Extra router work costs tens of microseconds per layer. Enable it
        // for sustained disk waits, and stop once RAM covers the hot routes.
        // Hysteresis avoids switching for individual fast/slow layers.
        const bool enabled=ssd_prediction_auto_?ssd_prediction_wait_ns_>=25000:ssd_prediction_wait_ns_>=50000;
        bool selected=enabled;
        if(ssd_prediction_cooldown_){--ssd_prediction_cooldown_;selected=false;}
        const bool sample=++ssd_prediction_observations_%48==0;
        if(selected && (!ssd_prediction_auto_ || sample)) {
            const auto host=host_experts_->stats();
            const auto started=host.prediction_started_jobs-ssd_prediction_probe_started_;
            const auto used=host.prediction_used_jobs-ssd_prediction_probe_used_;
            if(ssd_prediction_auto_ && started>=64 && used*4<started) {
                selected=false;ssd_prediction_cooldown_=48*64;++ssd_prediction_quality_rejections_;
            }
            if(!ssd_prediction_auto_ || started>=64) {
                ssd_prediction_probe_started_=host.prediction_started_jobs;
                ssd_prediction_probe_used_=host.prediction_used_jobs;
            }
        }
        if(selected!=ssd_prediction_auto_){ssd_prediction_auto_=selected;++ssd_prediction_switches_;}
    }
    bool ssd_router_prefetch() const noexcept {return ssd_prediction_auto_;}
    std::unique_ptr<SsdExpertReads> begin_ssd_expert_reads(std::vector<SsdExpertRequest> requests,std::size_t gate_up_requests);
    std::vector<MoeHostExpertCache::Lease> read_ssd_experts(const std::vector<SsdExpertRequest>& requests,bool background=false);
    struct SsdExpertCopy {MoeQuantRangeSource* source;int expert;std::vector<mfq_tensor_backend::Tensor> fields;};
    void copy_ssd_experts(const std::vector<SsdExpertCopy>& requests);
    std::future<void> queue_ssd_expert_copies(std::vector<SsdExpertCopy> requests);
    std::array<std::future<void>,2> queue_ssd_prefill_phases(
        std::array<std::vector<SsdExpertCopy>,2> requests);
    void calibrate_pipeline_pcie();
    bool complete_residency() const noexcept { return complete_residency_; }
    bool pipeline_ready() const noexcept { return complete_residency_ || ssd_experts_; }
    bool ssd_experts() const noexcept { return ssd_experts_; }
    bool suspend_residency(bool suspended) {
        if(suspended)finish_pipeline_exchanges();
        const bool previous=pipeline_residency_suspended_;
        pipeline_residency_suspended_=suspended;return previous;
    }
    bool set_reuse_learning(bool enabled) {
        const auto previous=pipeline_reuse_learning_;
        pipeline_reuse_learning_=enabled;
        if(host_experts_)host_experts_->set_reuse_learning(enabled);
        if(kv_expert_loan_)kv_expert_loan_->set_reuse_learning(enabled);
        return previous;
    }
    bool reuse_learning() const noexcept {return pipeline_reuse_learning_;}
    double ram_pcie_fraction() const noexcept { return ram_pcie_fraction_; }
    void finish_pipeline_exchanges() { if(moe_residency_)moe_residency_->apply_pending(); }
    void prepare_predictor_exchange() { if(moe_residency_)moe_residency_->prepare_predictor(); }
    void prepare_decode_residency() { if(moe_residency_)moe_residency_->prepare_decode(); }

    void begin_route_experts(
            const MoeRoutePlan & route,
            int n_experts) {
        if (route.host_unique_experts) return;
        const auto & ids = route.ids;
        if (!ids.is_cuda() || !ids.is_contiguous() ||
                ids.scalar_type() != mfq_tensor_backend::kInt32 ||
                ids.dim() != 2) {
            throw std::runtime_error(
                "cached MoE routes must be contiguous CUDA int32");
        }
        const int64_t count = ids.numel();
        if (count <= 0) {
            throw std::runtime_error(
                "cached MoE route list is empty");
        }
        if (n_experts <= 0) {
            throw std::runtime_error(
                "cached MoE expert count must be positive");
        }
        if (route_readback_pending_) {
            if (pending_route_generation_ == route.generation) return;
            throw std::runtime_error(
                "overlapping cached MoE route readbacks are unsupported");
        }
        if (!route_host_.defined() ||
                route_host_.numel() < count) {
            route_host_ = mfq_tensor_backend::empty(
                {count},
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCPU)
                    .dtype(mfq_tensor_backend::kInt32)
                    .pinned_memory(true));
        }
        auto current =
            mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(
            route_input_ready_, current));
        MFQ_CUDA_CHECK(cudaStreamWaitEvent(
            route_stream_, route_input_ready_, 0));
        const int64_t nbytes =
            count * static_cast<int64_t>(sizeof(int32_t));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            route_host_.data_ptr<int32_t>(),
            ids.data_ptr<int32_t>(),
            static_cast<size_t>(nbytes),
            cudaMemcpyDeviceToHost,
            route_stream_));
        MFQ_CUDA_CHECK(cudaEventRecord(
            route_done_, route_stream_));
        stats_.route_d2h_bytes += nbytes;
        pending_route_generation_ = route.generation;
        pending_route_count_ = count;
        pending_route_n_experts_ = n_experts;
        route_readback_pending_ = true;
    }

    std::vector<int32_t> read_route_experts(
            const MoeRoutePlan & route,
            int n_experts) {
        if (!route.host_unique_experts &&
                (!route_readback_pending_ ||
                 pending_route_generation_ != route.generation)) {
            begin_route_experts(route, n_experts);
        }
        if (!route_readback_pending_ ||
                pending_route_generation_ != route.generation ||
                pending_route_n_experts_ != n_experts) {
            throw std::runtime_error(
                "cached MoE route readback state does not match the route");
        }
        MFQ_CUDA_CHECK(cudaEventSynchronize(route_done_));
        const int64_t count = pending_route_count_;
        route_readback_pending_ = false;
        pending_route_generation_ = 0;
        pending_route_count_ = 0;
        pending_route_n_experts_ = 0;
        const auto * values =
            route_host_.data_ptr<int32_t>();
        std::vector<int32_t> result(
            values, values + count);
        std::sort(result.begin(), result.end());
        result.erase(
            std::unique(result.begin(), result.end()),
            result.end());
        for (int expert : result) {
            if (expert < 0 || expert >= n_experts) {
                throw std::runtime_error(
                    "MoE route selected an out-of-range expert");
            }
        }
        return result;
    }

    void record_compute_use() {
        MFQ_CUDA_CHECK(cudaEventRecord(
            compute_done_,
            mfq_get_current_cuda_stream().stream()));
        compute_done_recorded_ = true;
    }

    void count_full_projection_fallback() {
        ++stats_.full_projection_fallbacks;
    }
    void record_hybrid(int cpu, int gpu) {
        ++stats_.hybrid_calls;
        stats_.hybrid_cpu_experts += cpu;
        stats_.hybrid_gpu_experts += gpu;
    }
    void record_ram_pcie(int experts,std::int64_t bytes) {
        stats_.ram_pcie_experts+=experts; stats_.ram_pcie_bytes+=bytes;
    }

    void print_stats(std::ostream & stream) const {
        const auto host=host_experts_->stats();
        const auto kv_spec=kv_expert_loan_?kv_expert_loan_->speculative_stats():mfq::cuda::KvExpertLoan::SpeculativeStats{};
        const auto kv_reuse=kv_expert_loan_?kv_expert_loan_->reuse_stats():mfq::cuda::KvExpertLoan::ReuseStats{};
        stream << "moe_cache_stats"
               << " budget_bytes=" << budget_bytes_
               << " allocated_bytes=" << allocated_bytes_
               << " host_cache_budget_bytes=" << host.budget_bytes
               << " host_cache_resident_bytes=" << host.resident_bytes
               << " host_cache_managed_bytes=" << host.managed_bytes
               << " host_cache_managed_peak_bytes=" << host.managed_peak_bytes
               << " host_cache_transient_peak_bytes=" << host.transient_peak_bytes
               << " host_cache_hits=" << host.hits
               << " host_cache_misses=" << host.misses
               << " host_cache_evictions=" << host.evictions
               << " host_device_duplicate_evictions=" << host.device_duplicate_evictions
               << " host_device_duplicate_evicted_bytes=" << host.device_duplicate_evicted_bytes
               << " host_device_reuse_promotions=" << host.device_reuse_promotions
               << " host_speculative_windows=" << host.speculative_windows
               << " host_speculative_accepted_keys=" << host.speculative_accepted_keys
               << " host_cache_promotions=" << host.promotions
               << " host_cache_demotions=" << host.gpu_demotions
               << " complete_residency=" << complete_residency_
               << " pipeline_cpu_transfer_budget=" << pipeline_cpu_transfer_budget_
               << " pipeline_cpu_policy_changed_routes=" << pipeline_cpu_policy_changed_routes_
               << " pipeline_shared_cpu_cost_enabled=" << pipeline_shared_cpu_cost_
               << " pipeline_cpu_cost_keys=" << pipeline_cpu_cost_.keys()
               << " pipeline_cpu_cost_samples=" << pipeline_cpu_cost_.samples()
               << " pipeline_cpu_cost_queries=" << pipeline_cpu_cost_.queries()
               << " pipeline_cpu_cost_ready_queries=" << pipeline_cpu_cost_.ready_queries()
               << " pipeline_cpu_calibration_jobs=" << pipeline_cpu_calibration_.jobs()
               << " pipeline_cpu_calibration_observations=" << pipeline_cpu_calibration_.observations()
               << " pipeline_cpu_calibration_ns=" << uint64_t(pipeline_cpu_calibration_.work_ns())
               << " expert_disk_reads_after_preload=" << expert_disk_reads_after_preload()
#ifdef MFQ_NATIVE_CUDA_RUNTIME
               << " tensor_host_bytes=" << mfq::cuda::tensor_host_bytes.load()
               << " tensor_host_peak_bytes=" << mfq::cuda::tensor_host_peak_bytes.load()
#endif
               << " gpu_demote_bytes=" << stats_.gpu_demote_bytes
               << " host_bytes=" << host_bytes_
               << " demand_hits=" << stats_.demand_hits
               << " demand_misses=" << stats_.demand_misses
               << " prefetch_hits=" << stats_.prefetch_hits
               << " prefetch_misses=" << stats_.prefetch_misses
               << " evictions=" << stats_.evictions
               << " h2d_bytes=" << stats_.h2d_bytes
               << " route_d2h_bytes="
               << stats_.route_d2h_bytes
               << " full_projection_fallbacks="
               << stats_.full_projection_fallbacks
               << " hybrid_calls=" << stats_.hybrid_calls
               << " pipeline_transfer_cache_hits=" << stats_.pipeline_transfer_cache_hits
               << " pipeline_transfer_cache_misses=" << stats_.pipeline_transfer_cache_misses
               << " pipeline_transfer_cache_saved_bytes=" << stats_.pipeline_transfer_cache_saved_bytes
               << " hybrid_cpu_experts=" << stats_.hybrid_cpu_experts
               << " hybrid_gpu_experts=" << stats_.hybrid_gpu_experts
               << " ram_pcie_fraction=" << ram_pcie_fraction_
               << " ram_pcie_experts=" << stats_.ram_pcie_experts
               << " ram_pcie_bytes=" << stats_.ram_pcie_bytes
               << " pipeline_serves=" << stats_.pipeline_serves
               << " pipeline_prefill_cpu_copy_jobs=" << stats_.pipeline_prefill_cpu_copy_jobs
               << " pipeline_prefill_cpu_copy_join_ns=" << stats_.pipeline_prefill_cpu_copy_join_ns
               << " pipeline_two_stage_serves=" << stats_.pipeline_two_stage_serves
               << " pipeline_route_wait_ns=" << stats_.pipeline_route_wait_ns
               << " pipeline_plan_ns=" << stats_.pipeline_plan_ns
               << " pipeline_fetch_ns=" << stats_.pipeline_fetch_ns
               << " pipeline_ssd_read_ns=" << stats_.pipeline_ssd_read_ns
               << " pipeline_window_ssd_read_ns=" << stats_.pipeline_window_ssd_read_ns
               << " pipeline_cpu_ns=" << stats_.pipeline_cpu_ns
               << " pipeline_gpu_ns=" << stats_.pipeline_gpu_ns
               << " pipeline_gpu_interval_can_include_cpu_wait=" << (stats_.pipeline_two_stage_serves>0)
               << " pipeline_window_serves=" << stats_.pipeline_window_serves
               << " pipeline_window_route_ns=" << stats_.pipeline_window_route_ns
               << " pipeline_window_plan_ns=" << stats_.pipeline_window_plan_ns
               << " pipeline_window_fetch_ns=" << stats_.pipeline_window_fetch_ns
               << " pipeline_window_cpu_ns=" << stats_.pipeline_window_cpu_ns
               << " pipeline_window_gpu_ns=" << stats_.pipeline_window_gpu_ns
               << " pipeline_small_window_groups_1=" << stats_.pipeline_small_window_groups[1]
               << " pipeline_small_window_groups_2=" << stats_.pipeline_small_window_groups[2]
               << " pipeline_small_window_groups_3=" << stats_.pipeline_small_window_groups[3]
               << " pipeline_small_window_groups_4=" << stats_.pipeline_small_window_groups[4]
               << " pipeline_small_window_hot_groups_1=" << stats_.pipeline_small_window_hot_groups[1]
               << " pipeline_small_window_hot_groups_2=" << stats_.pipeline_small_window_hot_groups[2]
               << " pipeline_small_window_hot_groups_3=" << stats_.pipeline_small_window_hot_groups[3]
               << " pipeline_small_window_hot_groups_4=" << stats_.pipeline_small_window_hot_groups[4]
               << " pipeline_dma_copies=" << stats_.pipeline_dma_copies
               << " pipeline_window_dma_copies=" << stats_.pipeline_window_dma_copies
               << " pipeline_window_dma_bytes=" << stats_.pipeline_window_dma_bytes
               << " pipeline_window_pcie_payload_bytes=" << stats_.pipeline_window_pcie_payload_bytes
               << " pipeline_fused_transfer_copy_serves=" << stats_.pipeline_fused_transfer_copy_serves
               << " pipeline_empty_transfer_serves=" << stats_.pipeline_empty_transfer_serves
               << " kv_expert_loan_hit_bytes=" << (kv_expert_loan_?kv_expert_loan_->hits():0)
               << " kv_expert_loan_retired_bytes=" << (kv_expert_loan_?kv_expert_loan_->retired_bytes():0)
               << " kv_expert_reuse_promotions=" << kv_reuse.promotions
               << " kv_expert_reuse_enabled=" << kv_reuse.enabled
               << " kv_expert_reuse_hits=" << kv_reuse.protected_hits
               << " kv_expert_probation_evictions=" << kv_reuse.probation_evictions
               << " kv_expert_protected_pages=" << kv_reuse.protected_pages
               << " pipeline_dma_graph_launches=" << stats_.pipeline_dma_graph_launches
               << " pipeline_dma_graph_fallbacks=" << stats_.pipeline_dma_graph_fallbacks
               << " pipeline_ram_registered_bytes=" << pipeline_ram_registered_bytes_
               << " pipeline_prefill_group_serves=" << stats_.pipeline_prefill_group_serves
               << " pipeline_prefill_groups=" << stats_.pipeline_prefill_groups
               << " pipeline_prefill_ring_bytes=" << (pipeline_prefill_ring_.defined()?pipeline_prefill_ring_.numel():0)
               << " pipeline_direct_ram_bytes=" << stats_.pipeline_direct_ram_bytes
               << " pipeline_direct_ram_copies=" << stats_.pipeline_direct_ram_copies
               << " pipeline_staged_ram_bytes=" << stats_.pipeline_staged_ram_bytes
               << " pipeline_mapped_copy=" << pipeline_mapped_copy_
               << " pipeline_mapped_copy_serves=" << stats_.pipeline_mapped_copy_serves
               << " pipeline_mapped_overlap_serves=" << stats_.pipeline_mapped_overlap_serves
               << " pipeline_phased_transfer_serves=" << stats_.pipeline_phased_transfer_serves
               << " pipeline_gate_up_primary_down_missing_experts=" << stats_.pipeline_gate_up_primary_down_missing_experts
               << " pipeline_gate_up_primary_down_missing_positions=" << stats_.pipeline_gate_up_primary_down_missing_positions
               << " pipeline_early_gate_up_positions=" << stats_.pipeline_early_gate_up_positions
               << " pipeline_early_gate_up_enabled=" << stats_.pipeline_early_gate_up_enabled
               << " pipeline_early_no_cpu_ready_serves=" << stats_.pipeline_early_no_cpu_ready_serves
               << " pipeline_gpu_only_cpu_wait_skips=" << stats_.pipeline_gpu_only_cpu_wait_skips
               << " pipeline_fixed_gpu_timing_skips=" << stats_.pipeline_fixed_gpu_timing_skips
               << " pipeline_transfer_stream_writes=" << stats_.pipeline_transfer_stream_writes
               << " pipeline_transfer_host_callbacks=" << stats_.pipeline_transfer_host_callbacks
               << " pipeline_transfer_event_polls=" << stats_.pipeline_transfer_event_polls
               << " pipeline_kv_direct_read_bytes=" << stats_.pipeline_kv_direct_read_bytes
               << " pipeline_kv_direct_read_fields=" << stats_.pipeline_kv_direct_read_fields
               << " pipeline_kv_projection_hits=" << stats_.pipeline_kv_projection_hits
               << " pipeline_kv_projection_misses=" << stats_.pipeline_kv_projection_misses
               << " pipeline_kv_projection_fallbacks=" << stats_.pipeline_kv_projection_fallbacks
               << " pipeline_kv_projection_small_hit_bytes=" << stats_.pipeline_kv_projection_small_hit_bytes
               << " pipeline_kv_projection_early_hits=" << stats_.pipeline_kv_projection_early_hits
               << " pipeline_kv_gpu_resident_experts=" << stats_.pipeline_kv_gpu_resident_experts
               << " pipeline_kv_gpu_resident_bytes=" << stats_.pipeline_kv_gpu_resident_bytes
               << " kv_mtp_lru_windows=" << kv_spec.windows
               << " kv_mtp_lru_kept=" << kv_spec.kept
               << " kv_mtp_lru_reverted_hits=" << kv_spec.reverted_hits
               << " kv_mtp_lru_cold_insertions=" << kv_spec.cold_insertions
               << " kv_mtp_lru_cancels=" << kv_spec.cancels
               << " kv_mtp_lru_disabled=" << kv_spec.disabled
               << " pipeline_kv_gpu_early_gu_experts=" << stats_.pipeline_kv_gpu_early_gu_experts
               << " pipeline_reader_fence_reuses=" << stats_.pipeline_reader_fence_reuses
               << " pipeline_host_packet_serves=" << stats_.pipeline_host_packet_serves
               << " pipeline_direct_packet_serves=" << stats_.pipeline_direct_packet_serves
               << " pipeline_direct_packet_batch_serves=" << stats_.pipeline_direct_packet_batch_serves
               << " pipeline_host_packet_descriptor_bytes=" << stats_.pipeline_host_packet_descriptor_bytes
               << " pipeline_cached_phase_publishes=" << stats_.pipeline_cached_phase_publishes
               << " pipeline_kv_early_gu_serves=" << stats_.pipeline_kv_early_gu_serves
               << " pipeline_gu_build_overlap_serves=" << stats_.pipeline_gu_build_overlap_serves
               << " pipeline_gu_build_overlap_copies=" << stats_.pipeline_gu_build_overlap_copies
               << " pipeline_ssd_gu_overlap_serves=" << stats_.pipeline_ssd_gu_overlap_serves
               << " pipeline_ssd_gu_overlap_bytes=" << stats_.pipeline_ssd_gu_overlap_bytes
               << " pipeline_ssd_gu_staged_bytes=" << stats_.pipeline_ssd_gu_staged_bytes
               << " pipeline_ssd_gu_upload_batches=" << stats_.pipeline_ssd_gu_upload_batches
               << " pipeline_ssd_down_overlap_bytes=" << stats_.pipeline_ssd_down_overlap_bytes
               << " pipeline_ssd_down_upload_batches=" << stats_.pipeline_ssd_down_upload_batches
               << " pipeline_ssd_gu_compute_serves=" << stats_.pipeline_ssd_gu_compute_serves
               << " pipeline_native_dma_batches=" << stats_.pipeline_native_dma_batches
               << " pipeline_native_dma_copies=" << stats_.pipeline_native_dma_copies
               << " pipeline_native_dma_fallbacks=" << stats_.pipeline_native_dma_fallbacks
               << " pipeline_window_input_view_serves=" << stats_.pipeline_window_input_view_serves
               << " pipeline_gate_up_dma_bytes=" << stats_.pipeline_gate_up_dma_bytes
               << " pipeline_down_dma_bytes=" << stats_.pipeline_down_dma_bytes
               << " pipeline_prefetch_experts=" << stats_.pipeline_prefetch_experts
               << " pipeline_prefetch_bytes=" << stats_.pipeline_prefetch_bytes
               << " pipeline_prefetch_hit_bytes=" << stats_.pipeline_prefetch_hit_bytes
               << " pipeline_prefetch_busy_skips=" << stats_.pipeline_prefetch_busy_skips
               << " pipeline_mapped_copy_bytes=" << stats_.pipeline_mapped_copy_bytes
               << " pipeline_mapped_copy_descriptors=" << stats_.pipeline_mapped_copy_descriptors
               << " pipeline_wire_gpu_bytes=" << tensor_nbytes(pipeline_wire_gpu_)
               << " pipeline_host_stage_bytes=" << tensor_nbytes(pipeline_host_stage_)
               << " pipeline_sigmoid_table_bytes=" << tensor_nbytes(pipeline_sigmoid_table_)
               << " pipeline_wire_route_bytes=" << pipeline_wire_route_bytes_
               << " pipeline_wire_route_fields=" << pipeline_wire_route_fields_
               << " residency_adapt=" << config_.moe_residency_adapt
               << " residency_warm=" << config_.moe_residency_warm
               << " residency_projection_heat=" << (config_.moe_residency_warm && config_.moe_residency_projection_heat)
               << " residency_observed_routes=" << (moe_residency_?moe_residency_->stats().observed_routes:0)
               << " residency_rounds=" << (moe_residency_?moe_residency_->stats().scheduled_rounds:0)
               << " residency_candidates=" << (moe_residency_?moe_residency_->stats().candidate_bundles:0)
               << " residency_staging_budget_skips=" << (moe_residency_?moe_residency_->stats().staging_budget_skips:0)
               << " residency_plan_ns=" << (moe_residency_?moe_residency_->stats().plan_ns:0)
               << " residency_decode_plan_ns=" << (moe_residency_?moe_residency_->stats().decode_plan_ns:0)
               << " residency_speculative_windows=" << (moe_residency_?moe_residency_->stats().speculative_route_windows:0)
               << " residency_speculative_committed_routes=" << (moe_residency_?moe_residency_->stats().speculative_routes_committed:0)
               << " residency_speculative_heat_routes=" << (moe_residency_?moe_residency_->stats().speculative_heat_routes:0)
               << " residency_speculative_discarded_routes=" << (moe_residency_?moe_residency_->stats().speculative_routes_discarded:0)
               << " residency_speculative_cancels=" << (moe_residency_?moe_residency_->stats().speculative_route_cancels:0)
               << " residency_shadow_backup_bytes=" << (moe_residency_?moe_residency_->stats().shadow_backup_bytes:0)
               << " residency_shadow_backup_fields=" << (moe_residency_?moe_residency_->stats().shadow_backup_fields:0)
               << " residency_shadow_index_rebuilds=" << (moe_residency_?moe_residency_->stats().shadow_index_rebuilds:0)
               << " residency_shadow_index_matches=" << (moe_residency_?moe_residency_->stats().shadow_index_matches:0)
               << " residency_shadow_index_backup_bytes=" << (moe_residency_?moe_residency_->stats().shadow_index_backup_bytes:0)
               << " residency_planned_bundles=" << (moe_residency_?moe_residency_->stats().planned_bundles:0)
               << " residency_committed_bundles=" << (moe_residency_?moe_residency_->stats().committed_bundles:0)
               << " residency_memory_rejections=" << (moe_residency_?moe_residency_->stats().memory_rejections:0)
               << " residency_host_pack_bytes=" << (moe_residency_?moe_residency_->stats().host_pack_bytes:0)
               << " residency_direct_upload_bytes=" << (moe_residency_?moe_residency_->stats().direct_upload_bytes:0)
               << " residency_cached_copy_bytes=" << (moe_residency_?moe_residency_->stats().cached_copy_bytes:0)
               << " residency_cached_copy_fields=" << (moe_residency_?moe_residency_->stats().cached_copy_fields:0)
               << " residency_expert_fence_waits=" << (moe_residency_?moe_residency_->stats().expert_fence_waits:0)
               << " residency_cached_copy_batches=" << (moe_residency_?moe_residency_->stats().cached_copy_batches:0)
               << " residency_cached_copy_descriptors=" << (moe_residency_?moe_residency_->stats().cached_copy_descriptors:0)
               << " residency_cached_copy_blocks=" << (moe_residency_?moe_residency_->stats().cached_copy_blocks:0)
               << " residency_exchange_defaults=" << (moe_residency_?moe_residency_->stats().exchange_defaults:0)
               << " residency_kv_refill_bytes=" << (moe_residency_?moe_residency_->stats().kv_refill_bytes:0)
               << " residency_native_dma_batches=" << (moe_residency_?moe_residency_->stats().native_dma_batches:0)
               << " residency_native_dma_copies=" << (moe_residency_?moe_residency_->stats().native_dma_copies:0)
               << " residency_native_dma_fallbacks=" << (moe_residency_?moe_residency_->stats().native_dma_fallbacks:0)
               << " residency_expert_fence_wait_ns=" << (moe_residency_?moe_residency_->stats().expert_fence_wait_ns:0)
               << " residency_backup_submit_ns=" << (moe_residency_?moe_residency_->stats().backup_submit_ns:0)
               << " residency_refill_submit_ns=" << (moe_residency_?moe_residency_->stats().refill_submit_ns:0)
               << " residency_cached_copy_dma_bytes=" << (moe_residency_?moe_residency_->stats().cached_copy_dma_bytes:0)
               << " residency_cached_copy_dma_fields=" << (moe_residency_?moe_residency_->stats().cached_copy_dma_fields:0)
               << " residency_direct_exchange_rounds=" << (moe_residency_?moe_residency_->stats().direct_exchange_rounds:0)
               << " residency_direct_exchange_bytes=" << (moe_residency_?moe_residency_->stats().direct_exchange_bytes:0)
               << " residency_direct_exchange_ns=" << (moe_residency_?moe_residency_->stats().direct_exchange_ns:0)
               << " residency_backup_peak_bytes=" << (moe_residency_?moe_residency_->stats().backup_peak_bytes:0)
               << " residency_shared_backup_bytes=" << (moe_residency_?moe_residency_->stats().shared_backup_bytes:0)
               << " residency_candidate_projections=" << (moe_residency_?moe_residency_->stats().candidate_projections:0)
               << " residency_planned_projections=" << (moe_residency_?moe_residency_->stats().planned_projections:0)
               << " residency_committed_projections=" << (moe_residency_?moe_residency_->stats().committed_projections:0)
               << " residency_projection_memory_rejections=" << (moe_residency_?moe_residency_->stats().projection_memory_rejections:0)
               << " residency_prepare_ns=" << (moe_residency_?moe_residency_->stats().prepare_ns:0)
               << " residency_join_ns=" << (moe_residency_?moe_residency_->stats().join_ns:0)
               << " residency_dma_wait_ns=" << (moe_residency_?moe_residency_->stats().dma_wait_ns:0)
               << " residency_map_ns=" << (moe_residency_?moe_residency_->stats().map_ns:0)
               << " residency_host_exchange_ns=" << (moe_residency_?moe_residency_->stats().host_exchange_ns:0)
               << " residency_apply_ns=" << (moe_residency_?moe_residency_->stats().apply_ns:0)
               << " residency_decode_prepare_ns=" << (moe_residency_?moe_residency_->stats().decode_prepare_ns:0)
               << " residency_decode_join_ns=" << (moe_residency_?moe_residency_->stats().decode_join_ns:0)
               << " residency_decode_dma_wait_ns=" << (moe_residency_?moe_residency_->stats().decode_dma_wait_ns:0)
               << " residency_decode_map_ns=" << (moe_residency_?moe_residency_->stats().decode_map_ns:0)
               << " residency_decode_host_exchange_ns=" << (moe_residency_?moe_residency_->stats().decode_host_exchange_ns:0)
               << " residency_decode_apply_ns=" << (moe_residency_?moe_residency_->stats().decode_apply_ns:0)
               << " residency_backup_copies=" << (moe_residency_?moe_residency_->stats().backup_copies:0)
               << " residency_upload_copies=" << (moe_residency_?moe_residency_->stats().upload_copies:0)
               << " residency_parallel_host=" << (moe_residency_?moe_residency_->stats().parallel_host:0)
               << " residency_parallel_host_bytes=" << (moe_residency_?moe_residency_->stats().parallel_host_bytes:0)
               << " residency_parallel_host_batches=" << (moe_residency_?moe_residency_->stats().parallel_host_batches:0)
               << " residency_batched=" << (moe_residency_?moe_residency_->stats().batched:0)
               << " residency_batch_device_bytes=" << (moe_residency_?moe_residency_->stats().batch_device_bytes:0)
               << " residency_batch_descriptor_copies=" << (moe_residency_?moe_residency_->stats().batch_descriptor_copies:0)
               << " residency_async_publish=" << (moe_residency_?moe_residency_->stats().async_publish:0)
               << " residency_async_publish_ns=" << (moe_residency_?moe_residency_->stats().async_publish_ns:0)
               << " residency_decode_async_publish_ns=" << (moe_residency_?moe_residency_->stats().decode_async_publish_ns:0)
               << " residency_window_prepares=" << (moe_residency_?moe_residency_->stats().window_prepares:0)
               << " pipeline_adaptive=" << !config_.moe_ram_pcie_fraction.has_value()
               << " pipeline_dispatch_replay_used=" << pipeline_dispatch_replay_cursor_
               << " pipeline_dispatch_replay_total=" << pipeline_dispatch_replay_.size()
               << " h2d_submissions="
               << stats_.h2d_submissions
               << " h2d_descriptors="
               << stats_.h2d_descriptors
               << " mapped_registered_bytes="
               << mapped_registered_bytes_
               << " mapped_gather_bytes="
               << stats_.mapped_gather_bytes
               << " mapped_gather_submissions="
               << stats_.mapped_gather_submissions
               << " mapped_gather_descriptors="
               << stats_.mapped_gather_descriptors
               << " range_read_bytes="
               << stats_.range_read_bytes
               << " range_read_calls="
               << stats_.range_read_calls
               << " range_file_opens="
               << stats_.range_file_opens
               << " range_io_workers="
               << range_read_pool_->workers()
               << " range_read_ms="
               << static_cast<double>(stats_.range_read_nanoseconds) /
                    1.0e6
               << " range_overlap_batches="
               << stats_.range_overlap_batches
               << " range_overlap_wait_ms="
               << static_cast<double>(
                    stats_.range_overlap_wait_nanoseconds) / 1.0e6
               << "\n";
        if(moe_residency_)moe_residency_->trace_rank_summary(stream);
        print_cpu_profiles(stream);
        print_dma_profiles(stream);
        if(!pipeline_dispatch_record_path_.empty()) {
            std::ofstream file(pipeline_dispatch_record_path_);
            if(file)for(const auto& row:pipeline_dispatch_records_) {
                file<<row.layer<<' '<<row.tokens<<' '<<row.ids.size()<<' '<<row.cpu.size();
                for(const auto id:row.ids)file<<' '<<id;
                for(const auto id:row.cpu)file<<' '<<id;
                file<<'\n';
            }
        }
        if(!pipeline_miss_record_path_.empty()) {
            std::ofstream file(pipeline_miss_record_path_);
            if(!file)throw std::runtime_error("cannot write required MFQ expert miss audit");
            for(const auto& row:pipeline_dispatch_records_)if(!row.missing_bytes.empty()) {
                file<<row.layer<<' '<<row.tokens<<' '<<row.ids.size()<<' '<<row.cpu.size()<<' '<<row.missing_bytes.size();
                for(const auto id:row.ids)file<<' '<<id;
                for(const auto id:row.cpu)file<<' '<<id;
                for(const auto bytes:row.missing_bytes)file<<' '<<bytes;
                file<<'\n';
            }
            if(!file)throw std::runtime_error("required MFQ expert miss audit write failed");
        }
    }

    std::vector<std::pair<std::string, double>> memory_metrics() const {
        const auto host = host_experts_->stats();
        const auto kv_reuse=kv_expert_loan_?kv_expert_loan_->reuse_stats():mfq::cuda::KvExpertLoan::ReuseStats{};
        const auto residency=moe_residency_?moe_residency_->stats():MoeResidencyManager::Stats{};
        const auto [pinned_bytes,pinned_used]=pinned_expert_memory();
        const auto cached=host.managed_bytes+host.transient_bytes;
        const auto resident = std::size_t(std::max<int64_t>(0, host_bytes_)) + pinned_bytes+
            (cached>pinned_used?cached-pinned_used:0);
        const auto disk_reads=expert_disk_reads_after_preload();
        const auto disk_read_bytes=expert_disk_bytes_after_preload();
        const auto disk_times=expert_disk_timings();
        const auto direct_reads=expert_direct_field_reads();
        std::size_t gpu_weights=0;
        for(const auto& [signature,arena]:arenas_)
            for(const auto& field:arena->fields)gpu_weights+=std::size_t(tensor_nbytes(field));
        std::size_t transfer_bytes=0;
        for(const auto& [signature,arena]:pipeline_stages_)if(!signature.starts_with("prefill:")) {
            if(arena->packed.defined())transfer_bytes+=std::size_t(tensor_nbytes(arena->packed));
            for(const auto& field:arena->fields)transfer_bytes+=std::size_t(tensor_nbytes(field));
        }
        const auto cold=ssd_expert_total_bytes_>gpu_weights?ssd_expert_total_bytes_-gpu_weights:0;
        std::vector<std::pair<std::string,double>> result{{"ssd_expert_enabled", ssd_experts_ ? 1.0 : 0.0},
            {"ssd_io_workers",double(config_.moe_ssd_io_workers)},
            {"host_tensor_bytes",double(mfq::cuda::tensor_host_bytes.load())},
            {"ssd_expert_payload_bytes",double(ssd_experts_ && cold>host.resident_bytes?cold-host.resident_bytes:0)},
            {"ssd_expert_bytes_read",double(disk_read_bytes)},
            {"ssd_expert_read_count",double(disk_reads)},
            {"ssd_direct_field_reads",double(direct_reads)},
            {"ssd_bulk_field_reads",double(expert_bulk_field_reads())},
            {"ssd_ram_read_fast_hits",double(ssd_ram_read_fast_hits_.load())},
            {"ssd_scheduled_read_jobs",double(ssd_scheduled_read_jobs_.load())},
            {"ssd_reuse_promotions",double(kv_reuse.promotions)},
            {"ssd_reuse_policy_enabled",kv_reuse.enabled?1.0:0.0},
            {"ssd_ram_reuse_policy_enabled",host.reuse_priority?1.0:0.0},
            {"ssd_ram_reuse_promotions",double(host.reuse_promotions)},
            {"ssd_ram_scan_admissions",double(host.scan_admissions)},
            {"ssd_ram_scan_admitted_bytes",double(host.scan_admitted_bytes)},
            {"ssd_prefill_group_size",double(pipeline_prefill_group_capacity_)},
            {"ssd_prefill_predictor",pipeline_prefill_predictor_?1.0:0.0},
            {"ssd_ram_protected_bytes",double(host.reuse_protected_bytes)},
            {"ssd_reuse_hits",double(kv_reuse.protected_hits)},
            {"ssd_probation_evictions",double(kv_reuse.probation_evictions)},
            {"ssd_protected_cache_bytes",double(kv_reuse.protected_pages*mfq::cuda::KvExpertLoan::page)},
            {"ssd_reuse_capacity_bytes",double(kv_reuse.capacity_pages*mfq::cuda::KvExpertLoan::page)},
            {"ssd_reuse_protected_limit_bytes",double(kv_reuse.protected_limit_pages*mfq::cuda::KvExpertLoan::page)},
            {"ssd_reuse_protected_trims",double(kv_reuse.protected_trims)},
            {"ssd_reuse_protected_accesses",double(kv_reuse.protected_accesses)},
            {"ssd_reuse_accepted_row_promotions",double(kv_reuse.accepted_row_promotions)},
            {"ssd_decode_read_wait_ms",double(stats_.pipeline_window_ssd_read_ns)/1e6},
            {"ssd_expert_read_wait_ms",double(stats_.pipeline_ssd_read_ns)/1e6},
            {"expert_route_wait_ms",double(stats_.pipeline_route_wait_ns)/1e6},
            {"expert_plan_ms",double(stats_.pipeline_plan_ns)/1e6},
            {"expert_host_fetch_ms",double(stats_.pipeline_fetch_ns)/1e6},
            {"expert_host_copy_ms",double(stats_.pipeline_host_copy_ns)/1e6},
            {"expert_transfer_cache_prepare_ms",kv_expert_loan_?kv_expert_loan_->lookup_us()/1e3:0.0},
            {"expert_graph_captures",double(stats_.pipeline_graph_captures)},
            {"expert_target_multi_route_positions",double(stats_.pipeline_multi_route_positions[0])},
            {"expert_target_multi_unique_routes",double(stats_.pipeline_multi_unique_routes[0])},
            {"expert_predictor_multi_route_positions",double(stats_.pipeline_multi_route_positions[1])},
            {"expert_predictor_multi_unique_routes",double(stats_.pipeline_multi_unique_routes[1])},
            {"expert_logical_transfer_bytes",double(stats_.ram_pcie_bytes)},
            {"ssd_expert_encoded_ms",double(disk_times[0])/1e6},
            {"ssd_expert_materialize_ms",double(disk_times[1])/1e6},
            {"ssd_expert_nint_materialize_ms",double(disk_times[2])/1e6},
            {"ssd_expert_nvq_materialize_ms",double(disk_times[3])/1e6},
            {"ssd_prefill_host_stage_bytes",double(tensor_nbytes(pipeline_prefill_host_ring_)+tensor_nbytes(pipeline_prefill_metadata_host_))},
            {"ram_expert_limit_bytes",double(host.budget_bytes)},
            {"ram_expert_pinned_bytes",double(pinned_bytes)},
            {"ram_expert_pinned_used_bytes",double(pinned_used)},
            {"ram_expert_cache_hits",double(host.hits)},
            {"ram_expert_cache_misses",double(host.misses)},
            {"ram_expert_cache_evictions",double(host.evictions)},
            {"ram_expert_device_duplicate_evictions",double(host.device_duplicate_evictions)},
            {"ram_expert_device_duplicate_evicted_bytes",double(host.device_duplicate_evicted_bytes)},
            {"ram_expert_device_reuse_promotions",double(host.device_reuse_promotions)},
            {"ram_expert_speculative_windows",double(host.speculative_windows)},
            {"ram_expert_speculative_accepted_keys",double(host.speculative_accepted_keys)},
            {"ram_expert_speculative_cancels",double(host.speculative_cancels)},
            {"ram_expert_reuse_clock_rotations",double(host.reuse_clock_rotations)},
            {"ram_expert_ghost_entries",double(host.ghost_entries)},
            {"ram_expert_ghost_promotions",double(host.ghost_promotions)},
            {"ram_expert_enabled", resident > 0 ? 1.0 : 0.0},
            {"resident_expert_cache_bytes",double(gpu_weights+(kv_expert_loan_?kv_expert_loan_->bytes():0))},
            {"expert_decode_transfer_bytes",double(transfer_bytes)},
            {"expert_projection_dma_merges",double(stats_.pipeline_projection_dma_merges)},
            {"expert_native_dma_batches",double(stats_.pipeline_native_dma_batches)},
            {"expert_native_dma_copies",double(stats_.pipeline_native_dma_copies)},
            {"expert_native_dma_fallbacks",double(stats_.pipeline_native_dma_fallbacks)},
            {"expert_direct_cache_capacity_bytes",double(tensor_nbytes(pipeline_direct_cache_storage_))},
            {"expert_decode_stage_budget_bytes",double(pipeline_direct_cache_budget_)},
            {"expert_transfer_wire_bytes",double(tensor_nbytes(pipeline_wire_gpu_))},
            {"expert_wire_fallback_cells",double(pipeline_wire_fallback_cells_)},
            {"expert_wire_fallback_bytes",double(pipeline_wire_fallback_bytes_)},
            {"expert_gu_build_overlap_serves",double(stats_.pipeline_gu_build_overlap_serves)},
            {"ssd_gu_io_overlap_serves",double(stats_.pipeline_ssd_gu_overlap_serves)},
            {"ssd_gu_io_overlap_bytes",double(stats_.pipeline_ssd_gu_overlap_bytes)},
            {"ssd_gu_staged_upload_bytes",double(stats_.pipeline_ssd_gu_staged_bytes)},
            {"ssd_gu_upload_batches",double(stats_.pipeline_ssd_gu_upload_batches)},
            {"ssd_down_io_overlap_bytes",double(stats_.pipeline_ssd_down_overlap_bytes)},
            {"ssd_down_staged_upload_bytes",double(stats_.pipeline_ssd_down_staged_bytes)},
            {"ssd_down_upload_batches",double(stats_.pipeline_ssd_down_upload_batches)},
            {"ssd_down_early_max_batch",double(stats_.pipeline_ssd_down_early_max_batch)},
            {"ssd_down_early_max_bytes",double(stats_.pipeline_ssd_down_early_max_bytes)},
            {"ssd_gu_compute_overlap_serves",double(stats_.pipeline_ssd_gu_compute_serves)},
            {"ssd_gate_up_read_wait_ms",double(stats_.pipeline_ssd_gate_up_wait_ns)/1e6},
            {"ssd_down_read_wait_ms",double(stats_.pipeline_ssd_down_wait_ns)/1e6},
            {"ssd_router_prefetch_batches",double(ssd_prediction_batches_)},
            {"ssd_router_prefetch_jobs",double(ssd_prediction_jobs_)},
            {"ssd_router_prefetch_bytes",double(ssd_prediction_bytes_)},
            {"ssd_router_prefetch_used_jobs",double(host.prediction_used_jobs)},
            {"ssd_router_prefetch_used_bytes",double(host.prediction_used_bytes)},
            {"ssd_router_prefetch_started_jobs",double(host.prediction_started_jobs)},
            {"ssd_router_prefetch_started_bytes",double(host.prediction_started_bytes)},
            {"ssd_ram_prediction_first_uses",double(host.prediction_first_uses)},
            {"ssd_router_prefetch_matched_jobs",double(ssd_prediction_matched_)},
            {"ssd_router_prefetch_ready_jobs",double(ssd_prediction_ready_)},
            {"ssd_router_prefetch_canceled_jobs",double(ssd_prediction_canceled_jobs_)},
            {"ssd_router_prefetch_canceled_bytes",double(ssd_prediction_canceled_bytes_)},
            {"ssd_router_predicted_routes",double(ssd_predicted_routes_)},
            {"ssd_router_prediction_hits",double(ssd_prediction_hits_)},
            {"ssd_router_prefetch_busy_skips",double(ssd_prediction_busy_skips_)},
            {"ssd_router_prefetch_io_deferrals",double(ssd_prediction_io_deferrals_.load())},
            {"ssd_router_prefetch_spare_capacity_jobs",double(ssd_prediction_spare_jobs_.load())},
            {"ssd_router_prefetch_io_wait_us",double(ssd_prediction_wait_ns_)/1000},
            {"ssd_router_prefetch_route_wait_ms",double(ssd_prediction_route_wait_ns_)/1e6},
            {"ssd_router_gpu_protected_fields",double(ssd_prediction_gpu_protected_fields_)},
            {"ssd_router_gpu_eviction_skips",double(kv_reuse.forecast_eviction_skips)},
            {"ssd_router_prefetch_suggested",double(ssd_prediction_auto_)},
            {"ssd_router_prefetch_switches",double(ssd_prediction_switches_)},
            {"ssd_router_prefetch_quality_rejections",double(ssd_prediction_quality_rejections_)},
            {"ssd_router_prefetch_cooldown_layers",double(ssd_prediction_cooldown_)},
            {"ram_expert_payload_bytes", double(resident)},
            {"ram_expert_optional_backup_bytes",double(host.optional_bytes)},
              {"ram_expert_optional_reclaimed_bytes",double(host.optional_reclaimed_bytes)},
              {"residency_shadow_backup_bytes",double(residency.shadow_backup_bytes)},
              {"residency_shadow_backup_fields",double(residency.shadow_backup_fields)},
              {"residency_shadow_index_rebuilds",double(residency.shadow_index_rebuilds)},
              {"residency_shadow_index_matches",double(residency.shadow_index_matches)},
              {"residency_shadow_index_backup_bytes",double(residency.shadow_index_backup_bytes)},
              {"residency_scheduled_rounds",double(residency.scheduled_rounds)},
              {"residency_predictor_overlap_rounds",double(residency.predictor_overlap_rounds)},
              {"residency_predictor_overlap_serves",double(residency.predictor_overlap_serves)},
              {"residency_committed_projections",double(residency.committed_projections)},
            {"residency_ssd_load_requests",double(residency.ssd_load_requests)},
            {"residency_ssd_gpu_image_bytes",double(residency.ssd_gpu_image_bytes)},
            {"residency_ssd_gpu_images",double(residency.ssd_gpu_images)},
            {"residency_ssd_gpu_image_ms",double(residency.ssd_gpu_image_ns)/1e6},
            {"residency_ssd_gpu_image_copies",double(residency.ssd_gpu_image_copies)},
              {"residency_ssd_load_bytes",double(residency.ssd_load_bytes)},
              {"residency_kv_refill_bytes",double(residency.kv_refill_bytes)},
              {"residency_cached_copy_bytes",double(residency.cached_copy_bytes)},
              {"residency_direct_upload_bytes",double(residency.direct_upload_bytes)},
              {"residency_native_dma_batches",double(residency.native_dma_batches)},
              {"residency_native_dma_copies",double(residency.native_dma_copies)},
              {"residency_exchange_bytes",double(residency.direct_exchange_bytes)},
              {"residency_decode_plan_ms",double(residency.decode_plan_ns)/1e6},
              {"residency_decode_join_ms",double(residency.decode_join_ns)/1e6},
              {"residency_decode_dma_wait_ms",double(residency.decode_dma_wait_ns)/1e6},
              {"residency_speculative_committed_routes",double(residency.speculative_routes_committed)},
              {"residency_speculative_heat_routes",double(residency.speculative_heat_routes)},
              {"decode_window_pcie_payload_bytes",double(stats_.pipeline_window_pcie_payload_bytes)},
              {"decode_window_serves",double(stats_.pipeline_window_serves)},
              {"prefill_cpu_copy_jobs",double(stats_.pipeline_prefill_cpu_copy_jobs)},
              {"small_window_groups_1",double(stats_.pipeline_small_window_groups[1])},
              {"small_window_groups_2",double(stats_.pipeline_small_window_groups[2])},
              {"small_window_groups_3",double(stats_.pipeline_small_window_groups[3])},
              {"small_window_groups_4",double(stats_.pipeline_small_window_groups[4])},
              {"small_window_hot_groups_1",double(stats_.pipeline_small_window_hot_groups[1])},
              {"small_window_hot_groups_2",double(stats_.pipeline_small_window_hot_groups[2])},
              {"small_window_hot_groups_3",double(stats_.pipeline_small_window_hot_groups[3])},
              {"small_window_hot_groups_4",double(stats_.pipeline_small_window_hot_groups[4])},
              {"prefill_cpu_copy_join_ms",double(stats_.pipeline_prefill_cpu_copy_join_ns)/1e6},
#ifdef MFQ_NATIVE_CUDA_RUNTIME
            {"host_tensor_allocated_bytes",double(mfq::cuda::tensor_host_bytes.load())},
#endif
            {"ram_expert_pcie_read_bytes", double(stats_.ram_pcie_bytes)}};
        append_expert_format_metrics(result);
        return result;
    }

private:
    struct PrefillLoan {
        struct Arena {
            MoeGpuArena* arena;int first;std::vector<mfq_tensor_backend::Tensor> saved;
            MoeHostExpertCache::Lease owner;
            mfq_tensor_backend::Tensor keys;
            std::vector<std::pair<int,int>> refresh;
        };
        struct Cohort {std::vector<int32_t> experts;std::vector<int64_t> fields;};
        struct Copy {const uint8_t* source;std::size_t offset,bytes;};
        struct Layer {std::size_t bytes=0;std::vector<Copy> copies;};
        struct Stage {mfq_tensor_backend::Tensor host;bool pinned=false;};
        std::vector<Arena> arenas;
        std::map<std::pair<int,int>,Cohort> cohorts;
        std::map<int,Layer> layers;
        std::array<Stage,2> stages;
        std::size_t bytes=0;
        ~PrefillLoan() {for(auto& stage:stages)if(stage.pinned)(void)cudaHostUnregister(stage.host.data_ptr());}
    };
    std::unique_ptr<PrefillLoan> prefill_loan_;
    std::function<void(std::size_t)> reserve_ram_workspace_;
    struct DmaCallSample {
        int layer=0,tokens=0,entries=0;
        bool window=false;
        std::uint64_t serve=0,payload_bytes=0,copy_bytes=0,copies=0;
        std::uint64_t first_copy_bytes=0;
        std::uint64_t prepare_ns=0,enqueue_ns=0,fetch_ns=0,route_wait_ns=0,plan_ns=0;
        std::int64_t copy_ns=0,notice_ns=0,hot_ns=0;
        std::int64_t copy_begin_after_hot_ns=0,copy_end_after_hot_ns=0,notice_end_after_hot_ns=0;
        std::int64_t scatter_end_after_hot_ns=0;
    };
    bool pipeline_dma_profile_enabled_=false,pipeline_dma_profile_dropped_=false;
    bool pipeline_dma_profile_fail_alloc_=false;
    std::uint64_t pipeline_dma_profile_skipped_=0;
    std::vector<DmaCallSample> pipeline_dma_profiles_;
    void print_dma_profiles(std::ostream& stream) const {
        if(!pipeline_dma_profile_enabled_ && !pipeline_dma_profile_dropped_)return;
        // Only explicit producer-stream copies are observed. Mapped and phased
        // transfers use different completion protocols and are counted as skips.
        try {
            stream<<"moe_dma_profile enabled="<<pipeline_dma_profile_enabled_
                <<" dropped="<<pipeline_dma_profile_dropped_<<" samples="<<pipeline_dma_profiles_.size()
                <<" skipped="<<pipeline_dma_profile_skipped_<<'\n';
            for(const auto& p:pipeline_dma_profiles_)
                stream<<"moe_dma_call serve="<<p.serve<<" layer="<<p.layer
                    <<" tokens="<<p.tokens<<" entries="<<p.entries<<" window="<<p.window
                    <<" payload_bytes="<<p.payload_bytes<<" copy_bytes="<<p.copy_bytes<<" copies="<<p.copies
                    <<" first_copy_bytes="<<p.first_copy_bytes
                    <<" prepare_ns="<<p.prepare_ns<<" enqueue_ns="<<p.enqueue_ns<<" fetch_ns="<<p.fetch_ns
                    <<" route_wait_ns="<<p.route_wait_ns<<" plan_ns="<<p.plan_ns
                    <<" copy_ns="<<p.copy_ns<<" notice_ns="<<p.notice_ns<<" hot_ns="<<p.hot_ns
                    <<" copy_begin_after_hot_ns="<<p.copy_begin_after_hot_ns
                    <<" copy_end_after_hot_ns="<<p.copy_end_after_hot_ns
                    <<" notice_end_after_hot_ns="<<p.notice_end_after_hot_ns
                    <<" scatter_end_after_hot_ns="<<p.scatter_end_after_hot_ns<<'\n';
        } catch(const std::ios_base::failure&) {} catch(const std::bad_alloc&) {}
    }
    struct CpuProjectionSample {
        int expert=0,projection=0,family=0,format=0,bits=0,group_size=0;
        int positions=0,width=0,outputs=0;
        std::uint64_t work_ns=0,callbacks=0,rows=0;
    };
    struct CpuCallSample {
        int layer=0,tokens=0,experts=0,positions=0;
        bool window=false,activation_inside_gate_up=false;
        std::uint64_t serve=0,cell_call=0,prepare_ns=0,activation_ns=0,scatter_ns=0,total_ns=0;
        // Parallel work sum, already contained in the Gate/Up pool interval.
        std::uint64_t activation_work_ns=0;
        mfq::HostParallelProfile gate_up,down;
        std::vector<CpuProjectionSample> projections;
    };
    bool pipeline_cpu_profile_enabled_=false,pipeline_cpu_profile_dropped_=false;
    bool pipeline_cpu_profile_fail_alloc_=false;
    std::vector<CpuCallSample> pipeline_cpu_profiles_;
    void print_cpu_profiles(std::ostream& stream) const {
        if(!pipeline_cpu_profile_enabled_ && !pipeline_cpu_profile_dropped_)return;
        // Diagnostics are optional; a broken output does not change inference.
        try {
            stream<<"moe_cpu_profile enabled="<<pipeline_cpu_profile_enabled_
                  <<" dropped="<<pipeline_cpu_profile_dropped_<<" samples="<<pipeline_cpu_profiles_.size()<<'\n';
            for(const auto& sample:pipeline_cpu_profiles_) {
                stream<<"moe_cpu_call serve="<<sample.serve<<" cell_call="<<sample.cell_call
                    <<" layer="<<sample.layer<<" tokens="<<sample.tokens<<" window="<<sample.window
                    <<" experts="<<sample.experts<<" positions="<<sample.positions
                    <<" prepare_ns="<<sample.prepare_ns<<" activation_ns="<<sample.activation_ns
                    <<" activation_inside_gate_up="<<sample.activation_inside_gate_up
                    <<" activation_work_ns="<<sample.activation_work_ns
                    <<" scatter_ns="<<sample.scatter_ns<<" total_ns="<<sample.total_ns<<'\n';
                for(int stage=0;stage<2;++stage) {
                    const auto& p=stage==0 ? sample.gate_up : sample.down;
                    stream<<"moe_cpu_pool serve="<<sample.serve<<" stage="<<stage
                        <<" wall_ns="<<p.wall_ns<<" dispatch_wait_ns="<<p.dispatch_wait_ns
                        <<" setup_ns="<<p.setup_ns<<" work_ns="<<p.work_ns
                        <<" caller_work_ns="<<p.caller_work_ns<<" caller_wait_ns="<<p.caller_wait_ns
                        <<" wake_sum_ns="<<p.worker_wake_sum_ns<<" wake_max_ns="<<p.worker_wake_max_ns
                        <<" arrivals="<<p.worker_arrivals<<" participants="<<p.participants
                        <<" callbacks="<<p.callbacks<<" rows="<<p.rows<<" failed="<<p.failed<<'\n';
                }
                for(const auto& p:sample.projections)
                    stream<<"moe_cpu_projection serve="<<sample.serve<<" expert="<<p.expert
                        <<" projection="<<p.projection<<" family="<<p.family<<" format="<<p.format
                        <<" bits="<<p.bits<<" group_size="<<p.group_size<<" positions="<<p.positions
                        <<" width="<<p.width<<" outputs="<<p.outputs
                        <<" work_ns="<<p.work_ns<<" callbacks="<<p.callbacks<<" rows="<<p.rows<<'\n';
            }
        } catch(const std::ios_base::failure&) {} catch(const std::bad_alloc&) {}
    }
    std::uint64_t expert_disk_reads_after_preload() const;
    std::uint64_t expert_disk_bytes_after_preload() const;
    std::uint64_t expert_direct_field_reads() const;
    std::uint64_t expert_bulk_field_reads() const;
    std::pair<std::size_t,std::size_t> pinned_expert_memory() const;
    std::array<std::uint64_t,4> expert_disk_timings() const;
    void append_expert_format_metrics(std::vector<std::pair<std::string,double>>& metrics) const;
    bool complete_residency_=false;
    bool ssd_experts_=false;
    std::size_t ssd_expert_total_bytes_=0;
    SsdReadPriority ssd_read_priority_;
    std::unique_ptr<mfq::HostParallelPool> ssd_read_pool_;
    std::unique_ptr<mfq::HostParallelPool> ssd_background_read_pool_;
    std::shared_ptr<MoeSsdPrefillReader> ssd_prefill_reader_;
    std::unique_ptr<mfq::HostParallelPool> ssd_prediction_pool_;
    std::shared_ptr<MoeSsdPrefillReader> ssd_prediction_reader_;
    std::unique_ptr<SsdExpertReads> ssd_prediction_reads_,ssd_previous_prediction_;
    std::vector<SsdExpertRequest> ssd_pending_prediction_;
    std::function<void()> ssd_pending_forecast_;
    std::vector<mfq::cuda::KvExpertLoan::Key> ssd_forecast_gpu_fields_;
    std::uint64_t ssd_prediction_gpu_protected_fields_=0;
    std::uint64_t ssd_prediction_batches_=0,ssd_prediction_jobs_=0,ssd_prediction_bytes_=0,ssd_prediction_busy_skips_=0;
    std::uint64_t ssd_prediction_wait_ns_=0,ssd_prediction_switches_=0;
    std::uint64_t ssd_predicted_routes_=0,ssd_prediction_hits_=0;
    std::uint64_t ssd_prediction_matched_=0,ssd_prediction_ready_=0;
    std::uint64_t ssd_prediction_canceled_jobs_=0,ssd_prediction_canceled_bytes_=0;
    std::uint64_t ssd_prediction_observations_=0,ssd_prediction_cooldown_=0,ssd_prediction_quality_rejections_=0;
    std::uint64_t ssd_prediction_probe_started_=0,ssd_prediction_probe_used_=0;
    bool ssd_prediction_auto_=false;
    std::uint64_t ssd_prediction_route_wait_ns_=0;
    std::atomic<std::uint64_t> ssd_ram_read_fast_hits_{0},ssd_scheduled_read_jobs_{0};
    std::atomic<std::uint64_t> ssd_prediction_io_deferrals_{0},ssd_prediction_spare_jobs_{0};
    bool pipeline_residency_suspended_=false;
    bool pipeline_reuse_learning_=true;
    CudaExecutionConfig config_;
    double ram_pcie_fraction_=0;
    double pipeline_pcie_gbps_=0;
    bool pipeline_cpu_transfer_budget_=mfq::cuda::runtime_options::cpu_transfer_budget();
    bool pipeline_comparison_cpu_policy_=false;
    std::uint64_t pipeline_cpu_policy_changed_routes_=0;
    bool pipeline_shared_cpu_cost_=false;
    mfq::MoeCpuCostModel pipeline_cpu_cost_;
    mfq::MoeCpuCalibration pipeline_cpu_calibration_;
    bool measuring_preload_h2d_=false;
    std::vector<std::unique_ptr<MoePreloadH2DTiming>> preload_h2d_timings_;
    friend class MoeCachedSource;
    friend class MoeFfnPipeline;
    friend class MoeResidencyManager;
    friend void release_moe_prefill_buffers(MoeExpertCache* cache);
    friend void release_moe_decode_buffers(const std::shared_ptr<MoeExpertCache>& cache);
    friend bool lend_moe_decode_cache_for_prefill(const std::shared_ptr<MoeExpertCache>&,std::size_t);
    friend void begin_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>&,int);
    friend void commit_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>&,int);
    friend void discard_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>&) noexcept;
    friend bool prepare_moe_graph_pool(const std::shared_ptr<MoeExpertCache>& cache);
    friend struct MoeResidencyTestAccess;
    friend void prepare_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>&,bool);
    friend void prepare_moe_cpu_budget_comparison(const std::shared_ptr<MoeExpertCache>&,bool,bool);
    friend void prepare_moe_cpu_calibration_comparison(const std::shared_ptr<MoeExpertCache>&,bool);
    friend std::size_t finish_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>&);
    std::unique_ptr<MoeResidencyManager> moe_residency_;
    std::vector<std::array<int,3>> pipeline_bundles_;
    std::unordered_map<std::string,std::unique_ptr<MoeGpuArena>> pipeline_stages_;
    std::size_t pipeline_direct_cache_budget_=0;
    mfq_tensor_backend::Tensor pipeline_direct_cache_storage_;
    std::optional<std::size_t> pipeline_direct_cache_region_;
    uint64_t pipeline_wire_fallback_cells_=0,pipeline_wire_fallback_bytes_=0;
    std::vector<std::function<void()>> pipeline_decode_release_;
    std::vector<std::function<bool()>> pipeline_decode_indirect_;
    // Eager prefill projections execute in order on pipeline_pool_stream_.
    // Their quantization scratch can be reused across projections and layers.
    std::shared_ptr<MixedMoeRuntime> pipeline_prefill_activations_;
    std::shared_ptr<void> pipeline_prefill_dma_metadata_;
    std::shared_ptr<void> pipeline_prefill_route_publication_;
    mfq_tensor_backend::Tensor pipeline_prefill_ring_;
    int pipeline_prefill_group_capacity_=0;
    bool pipeline_prefill_predictor_=false;
    mfq_tensor_backend::Tensor pipeline_prefill_host_ring_;
    mfq_tensor_backend::Tensor pipeline_prefill_metadata_host_;
    std::size_t pipeline_prefill_metadata_stride_=0;
    std::size_t pipeline_prefill_host_stride_=0;
    bool pipeline_prefill_layer_layout_=false;
    std::map<int,std::pair<int64_t,int64_t>> pipeline_prefill_layers_;
    std::map<int,int64_t> pipeline_prefill_layer_gate_up_bytes_;
    mfq_tensor_backend::Tensor pipeline_prefill_layer_buffer_;
    mfq_tensor_backend::Tensor pipeline_prefill_layer_second_buffer_;
    mfq_tensor_backend::Tensor pipeline_prefill_layer_down_buffer_;
    int64_t pipeline_prefill_layer_split_=0;
    int64_t pipeline_prefill_layer_stride_=0;
    std::function<void(std::size_t)> reserve_workspace_;
    bool pipeline_prefill_layer_overlap_=false;
    std::array<cudaEvent_t,2> pipeline_prefill_layer_ready_{};
    std::array<cudaEvent_t,2> pipeline_prefill_layer_gate_up_ready_{};
    std::array<std::optional<int>,2> pipeline_prefill_layer_loaded_{};
    std::array<std::future<PrefillCopySubmission>,2> pipeline_prefill_copy_tasks_;
    std::array<std::optional<int>,2> pipeline_prefill_copy_targets_{};
    int pipeline_prefill_copy_pending_=0;
    bool pipeline_prefill_cpu_reported_=false;
    int pipeline_prefill_layer_last_served_=-1;
    std::array<cudaEvent_t,2> pipeline_prefill_layer_read_done_{};
    std::array<bool,2> pipeline_prefill_layer_read_recorded_{};
    cudaMemPool_t pipeline_prefill_pool_=nullptr;
    uint64_t pipeline_prefill_pool_previous_threshold_=0;
    std::vector<std::function<void()>> pipeline_prefill_release_;
    std::unordered_map<std::string,std::shared_ptr<MoeFfnPrefillHostWorkspace>> pipeline_prefill_host_workspaces_;
    bool pipeline_transfer_cache_=mfq::cuda::runtime_options::transfer_cache();
    bool pipeline_mapped_copy_=mfq::cuda::runtime_options::mapped_copy();
    std::vector<MoeCacheNewLease> pipeline_transfer_slots_;
    mfq_tensor_backend::Tensor pipeline_host_stage_;
    mfq_tensor_backend::Tensor pipeline_wire_gpu_;
    std::array<std::unique_ptr<mfq::cuda::DmaCopyBatch>,2> pipeline_dma_batches_;
    mfq_tensor_backend::Tensor pipeline_prefetch_wire_gpu_,pipeline_prefetch_host_stage_;
    std::vector<mfq::MoeCacheKey> pipeline_prefetched_keys_;
    mfq_tensor_backend::Tensor pipeline_sigmoid_table_;
    int64_t pipeline_wire_route_bytes_=0,pipeline_wire_route_fields_=0;
    std::array<std::pair<int64_t,int64_t>,2> pipeline_wire_geometry_{};
    std::unordered_map<int,std::vector<int64_t>> pipeline_host_staged_expert_bytes_;
    std::shared_ptr<mfq_tensor_backend::Tensor> pipeline_ram_complement_;
    std::shared_ptr<MoePinnedExpertArena> pinned_expert_arena_;
    int64_t pipeline_ram_registered_bytes_=0;
    void* pipeline_ram_alias_=nullptr;
    void* pipeline_host_stage_alias_=nullptr;
    std::vector<MoeHostExpertCache::Lease> pipeline_dma_leases_;
    struct DispatchRecord {int layer=0,tokens=0;std::vector<int32_t> ids,cpu;std::vector<uint64_t> missing_bytes;};
    std::string pipeline_miss_record_path_;
    std::string pipeline_dispatch_record_path_;
    bool pipeline_dispatch_capture_required_=false;
    std::vector<DispatchRecord> pipeline_comparison_records_;
    std::vector<DispatchRecord> pipeline_dispatch_records_,pipeline_dispatch_replay_;
    std::size_t pipeline_dispatch_replay_cursor_=0;
    std::shared_ptr<mfq::cuda::Context> pipeline_pool_context_;
    cudaStream_t pipeline_pool_stream_=nullptr;

    void append_source_transfers(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch,
        std::vector<MoeCacheTransfer> & transfers,
        bool & replaced_occupied,
        std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> * held_slots,
        std::vector<MoeCacheNewLease> * new_leases);

    void rollback_preparation(
        const std::vector<MoeCacheNewLease> & new_leases,
        const std::vector<
            std::pair<mfq::MoeCacheSlotBook *, int>> & held_slots) noexcept;

    bool begin_deferred_range_read(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts);

    void finish_deferred_range_read(MoeCachedSource & source);

    void record_range_read(
        const mfq::cuda::MfeMxfp4ReadBatchStats & range_stats) {
        stats_.range_read_bytes += static_cast<int64_t>(range_stats.bytes);
        stats_.range_read_calls += static_cast<int64_t>(range_stats.calls);
        stats_.range_file_opens +=
            static_cast<int64_t>(range_stats.file_opens);
        stats_.range_read_nanoseconds +=
            static_cast<int64_t>(range_stats.wall_nanoseconds);
    }

    MoePinnedStage & acquire_stage(
            int64_t required_bytes,
            bool require_device) {
        for (size_t offset = 0; offset < stages_.size(); ++offset) {
            const size_t index =
                (next_stage_ + offset) % stages_.size();
            auto & stage = stages_[index];
            if (!stage.pending ||
                cudaEventQuery(stage.done) == cudaSuccess) {
                stage.pending = false;
                next_stage_ = (index + 1) % stages_.size();
                if (!stage.host.defined() ||
                        stage.host.numel() < required_bytes) {
                    int64_t capacity = 1;
                    while (capacity < required_bytes) capacity *= 2;
                    stage.host = mfq_tensor_backend::empty(
                        {capacity},
                        mfq_tensor_backend::TensorOptions()
                            .device(mfq_tensor_backend::kCPU)
                            .dtype(mfq_tensor_backend::kUInt8)
                            .pinned_memory(true));
                }
                if (require_device &&
                        (!stage.device.defined() ||
                         stage.device.numel() < required_bytes)) {
                    int64_t capacity = 1;
                    while (capacity < required_bytes) capacity *= 2;
                    stage.device = mfq_tensor_backend::empty(
                        {capacity},
                        mfq_tensor_backend::TensorOptions()
                            .device(mfq_tensor_backend::kCUDA)
                            .dtype(mfq_tensor_backend::kUInt8));
                }
                return stage;
            }
            (void)cudaGetLastError();
        }
        auto & stage = stages_[next_stage_];
        MFQ_CUDA_CHECK(cudaEventSynchronize(stage.done));
        stage.pending = false;
        next_stage_ = (next_stage_ + 1) % stages_.size();
        if (!stage.host.defined() ||
                stage.host.numel() < required_bytes) {
            int64_t capacity = 1;
            while (capacity < required_bytes) capacity *= 2;
            stage.host = mfq_tensor_backend::empty(
                {capacity},
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCPU)
                    .dtype(mfq_tensor_backend::kUInt8)
                    .pinned_memory(true));
        }
        if (require_device &&
                (!stage.device.defined() ||
                 stage.device.numel() < required_bytes)) {
            int64_t capacity = 1;
            while (capacity < required_bytes) capacity *= 2;
            stage.device = mfq_tensor_backend::empty(
                {capacity},
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kUInt8));
        }
        return stage;
    }

    void submit_transfers(
            const std::vector<MoeCacheTransfer> & transfers,
            bool waits_for_compute,
            bool wait_on_compute_stream) {
        std::vector<mfq::cuda::MfeMxfp4ReadRequest> range_requests;
        auto materialize_source = [&range_requests](
                const MoeCacheTransfer & transfer,
                uint8_t * destination) {
            if (transfer.range_store != nullptr) {
                range_requests.push_back({
                    transfer.range_store,
                    transfer.range_part,
                    std::span<uint8_t>(
                        destination,
                        static_cast<size_t>(transfer.nbytes)),
                });
            } else {
                std::memcpy(
                    destination,
                    transfer.source,
                    static_cast<size_t>(transfer.nbytes));
            }
        };
        auto finish_range_reads = [this, &range_requests]() {
            if (range_requests.empty()) return;
            const auto range_stats = range_read_pool_->read(range_requests);
            record_range_read(range_stats);
            range_requests.clear();
        };
        int64_t staged_payload_bytes = 0;
        int transfer_count = 0;
        int staged_count = 0;
        int mapped_count = 0;
        for (const auto & transfer : transfers) {
            const bool range_source =
                transfer.range_store != nullptr &&
                transfer.range_part != nullptr;
            if (transfer.nbytes < 0 ||
                (transfer.nbytes > 0 &&
                 ((!range_source && transfer.source == nullptr) ||
                  transfer.destination == nullptr)) ||
                ((transfer.range_store == nullptr) !=
                 (transfer.range_part == nullptr)) ||
                (range_source &&
                 (transfer.mapped_source != nullptr ||
                  transfer.range_part->nbytes !=
                    static_cast<uint64_t>(transfer.nbytes)))) {
                throw std::runtime_error(
                    "invalid MoE cache transfer");
            }
            if (transfer.nbytes == 0) continue;
            const bool direct_mapped =
                !prewarming_ && !measuring_preload_h2d_ && transfer.mapped_source != nullptr;
            if (direct_mapped) {
                ++mapped_count;
            } else {
                staged_payload_bytes =
                    (staged_payload_bytes + 15) & ~int64_t{15};
                if (transfer.nbytes >
                        std::numeric_limits<int64_t>::max() -
                            staged_payload_bytes) {
                    throw std::overflow_error(
                        "MoE cache transfer byte count overflows int64");
                }
                staged_payload_bytes += transfer.nbytes;
                ++staged_count;
            }
            ++transfer_count;
        }
        if (transfer_count == 0) {
            if (wait_on_compute_stream &&
                    transfer_ready_recorded_) {
                MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                    mfq_get_current_cuda_stream().stream(),
                    transfer_ready_, 0));
            }
            return;
        }
        ++stats_.h2d_submissions;
        stats_.h2d_descriptors += transfer_count;
        if (prewarming_ || measuring_preload_h2d_) {
            auto & stage = acquire_stage(staged_payload_bytes, false);
            auto * staging = stage.host.data_ptr<uint8_t>();
            int64_t offset = 0;
            for (const auto & transfer : transfers) {
                if (transfer.nbytes == 0) continue;
                offset = (offset + 15) & ~int64_t{15};
                materialize_source(transfer, staging + offset);
                offset += transfer.nbytes;
            }
            finish_range_reads();
            MoePreloadH2DTiming* timing=nullptr;
            if(measuring_preload_h2d_) {
                preload_h2d_timings_.push_back(std::make_unique<MoePreloadH2DTiming>(staged_payload_bytes));
                timing=preload_h2d_timings_.back().get();
                MFQ_CUDA_CHECK(cudaEventRecord(timing->begin,weight_stream_));
            }
            offset = 0;
            for (const auto & transfer : transfers) {
                if (transfer.nbytes == 0) continue;
                offset = (offset + 15) & ~int64_t{15};
                MFQ_CUDA_CHECK(cudaMemcpyAsync(
                    transfer.destination,
                    staging + offset,
                    static_cast<size_t>(transfer.nbytes),
                    cudaMemcpyHostToDevice,
                    weight_stream_));
                if (transfer.packed_weight) {
                    stats_.h2d_bytes += transfer.nbytes;
                }
                offset += transfer.nbytes;
            }
            if(timing)MFQ_CUDA_CHECK(cudaEventRecord(timing->end,weight_stream_));
            MFQ_CUDA_CHECK(cudaEventRecord(
                stage.done, weight_stream_));
            MFQ_CUDA_CHECK(cudaEventRecord(
                transfer_ready_, weight_stream_));
            transfer_ready_recorded_ = true;
            stage.pending = true;
            if (wait_on_compute_stream) {
                MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                    mfq_get_current_cuda_stream().stream(),
                    transfer_ready_, 0));
            }
            return;
        }
        const int64_t scatter_descriptor_offset =
            (staged_payload_bytes + 15) & ~int64_t{15};
        const int64_t scatter_descriptor_bytes =
            static_cast<int64_t>(staged_count) *
            static_cast<int64_t>(
                sizeof(mfq::MoeCacheScatterDescriptor));
        if (scatter_descriptor_bytes >
                std::numeric_limits<int64_t>::max() -
                    scatter_descriptor_offset) {
            throw std::overflow_error(
                "MoE cache descriptor byte count overflows int64");
        }
        const int64_t mapped_descriptor_offset =
            (scatter_descriptor_offset +
             scatter_descriptor_bytes + 15) & ~int64_t{15};
        const int64_t mapped_descriptor_bytes =
            static_cast<int64_t>(mapped_count) *
            static_cast<int64_t>(
                sizeof(mfq::MoeCacheMappedCopyDescriptor));
        if (mapped_descriptor_bytes >
                std::numeric_limits<int64_t>::max() -
                    mapped_descriptor_offset) {
            throw std::overflow_error(
                "MoE cache mapped descriptor byte count overflows int64");
        }
        const int64_t total_bytes =
            mapped_descriptor_offset + mapped_descriptor_bytes;
        auto & stage = acquire_stage(total_bytes, true);
        auto * staging = stage.host.data_ptr<uint8_t>();
        auto * scatter_descriptors =
            reinterpret_cast<mfq::MoeCacheScatterDescriptor *>(
                staging + scatter_descriptor_offset);
        auto * mapped_descriptors =
            reinterpret_cast<mfq::MoeCacheMappedCopyDescriptor *>(
                staging + mapped_descriptor_offset);
        int64_t offset = 0;
        int scatter_descriptor = 0;
        int mapped_descriptor = 0;
        for (const auto & transfer : transfers) {
            if (transfer.nbytes == 0) continue;
            if (transfer.mapped_source != nullptr) {
                mapped_descriptors[mapped_descriptor++] = {
                    static_cast<uint64_t>(
                        reinterpret_cast<uintptr_t>(
                            transfer.destination)),
                    static_cast<uint64_t>(
                        reinterpret_cast<uintptr_t>(
                            transfer.mapped_source)),
                    static_cast<uint64_t>(transfer.nbytes),
                };
                stats_.mapped_gather_bytes += transfer.nbytes;
            } else {
                offset = (offset + 15) & ~int64_t{15};
                materialize_source(transfer, staging + offset);
                scatter_descriptors[scatter_descriptor++] = {
                    static_cast<uint64_t>(
                        reinterpret_cast<uintptr_t>(
                            transfer.destination)),
                    static_cast<uint64_t>(offset),
                    static_cast<uint64_t>(transfer.nbytes),
                };
                offset += transfer.nbytes;
            }
            if (transfer.packed_weight) {
                stats_.h2d_bytes += transfer.nbytes;
            }
        }
        finish_range_reads();
        if (scatter_descriptor != staged_count ||
                mapped_descriptor != mapped_count) {
            throw std::runtime_error(
                "MoE cache transfer descriptor count mismatch");
        }
        if (waits_for_compute && compute_done_recorded_) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                weight_stream_, compute_done_, 0));
        }
        MoePreloadH2DTiming* h2d_timing=nullptr;
        if (measuring_preload_h2d_) {
            preload_h2d_timings_.push_back(std::make_unique<MoePreloadH2DTiming>(total_bytes));
            h2d_timing=preload_h2d_timings_.back().get();
            MFQ_CUDA_CHECK(cudaEventRecord(h2d_timing->begin,weight_stream_));
        }
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            stage.device.data_ptr<uint8_t>(),
            staging,
            static_cast<size_t>(total_bytes),
            cudaMemcpyHostToDevice,
            weight_stream_));
        if (h2d_timing) MFQ_CUDA_CHECK(cudaEventRecord(h2d_timing->end,weight_stream_));
        if (staged_count > 0) {
            mfq::moe_cache_scatter_cuda(
                stage.device.data_ptr<uint8_t>(),
                scatter_descriptor_offset,
                staged_count,
                weight_stream_);
        }
        if (mapped_count > 0) {
            auto * device_descriptors =
                reinterpret_cast<const mfq::MoeCacheMappedCopyDescriptor *>(
                    stage.device.data_ptr<uint8_t>() +
                    mapped_descriptor_offset);
            mfq::moe_cache_mapped_gather_cuda(
                device_descriptors,
                mapped_count,
                mapped_copy_blocks_,
                weight_stream_);
            ++stats_.mapped_gather_submissions;
            stats_.mapped_gather_descriptors += mapped_count;
        }
        MFQ_CUDA_CHECK(cudaEventRecord(stage.done, weight_stream_));
        MFQ_CUDA_CHECK(cudaEventRecord(
            transfer_ready_, weight_stream_));
        transfer_ready_recorded_ = true;
        stage.pending = true;
        if (wait_on_compute_stream) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                mfq_get_current_cuda_stream().stream(),
                transfer_ready_, 0));
        }
    }

    void invalidate(const mfq::MoeCacheKey & key, int slot,bool retain_cold=true);
    void publish_quant_promotions(const std::vector<MoeCacheNewLease>& leases);

    int64_t budget_bytes_ = 0;
    int64_t allocated_bytes_ = 0;
    int64_t host_bytes_ = 0;
    bool finalized_ = false;
    bool prewarming_ = false;
    cudaStream_t weight_stream_ = nullptr;
    cudaStream_t route_stream_ = nullptr;
    cudaEvent_t compute_done_ = nullptr;
    bool compute_done_recorded_ = false;
    cudaEvent_t transfer_ready_ = nullptr;
    bool transfer_ready_recorded_ = false;
    cudaEvent_t route_input_ready_ = nullptr;
    cudaEvent_t route_done_ = nullptr;
    mfq_tensor_backend::Tensor route_host_;
    uint64_t pending_route_generation_ = 0;
    int64_t pending_route_count_ = 0;
    int pending_route_n_experts_ = 0;
    bool route_readback_pending_ = false;
    std::vector<MoePinnedStage> stages_;
    size_t next_stage_ = 0;
    std::unordered_map<std::string, std::unique_ptr<MoeGpuArena>> arenas_;
    std::vector<std::shared_ptr<MoeCachedSource>> sources_;
    std::vector<int> prefill_low_priority_layers_;
    std::optional<mfq::MoeCacheProfile> profile_;
    std::vector<mfq::MoeProfileCandidate> prewarm_selected_;
    MoeCacheStats stats_;
    std::shared_ptr<MoeHostExpertCache> host_experts_;
    struct RegisteredHostField {
        void * host = nullptr;
        int64_t bytes = 0;
        bool owned = false;
    };
    bool mapped_gather_enabled_ = false;
    int mapped_copy_blocks_ = 64;
    int64_t mapped_registered_bytes_ = 0;
    std::vector<RegisteredHostField> registered_host_fields_;
    std::unordered_map<void *, const uint8_t *> mapped_host_lookup_;
    std::unique_ptr<mfq::cuda::MfeMxfp4ReadPool> range_read_pool_;
    std::unordered_map<int, std::unique_ptr<MoePendingRangeRead>>
        pending_range_reads_;
    bool range_overlap_enabled_ = true;
};
