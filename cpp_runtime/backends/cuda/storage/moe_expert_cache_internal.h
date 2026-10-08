#include "../runtime/execution_options.h"
#pragma once

#include "moe_cache_types_internal.h"
#include "mfq_cuda_context.h"
#include "runtime/moe_residency.h"
#include "mfq/host_parallel.h"
#include "mfq/moe_cpu_cost_model.h"
#include "mfq/moe_cpu_calibration.h"
#include <array>
#include <cstring>
#include <fstream>

class MoeExpertCache {
public:
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
        pipeline_cpu_calibration_.wait();
        moe_residency_.reset();
        if(pipeline_pool_context_) {
            (void)cudaStreamSynchronize(pipeline_pool_stream_);
            pipeline_pool_context_->end_graph_pool(pipeline_pool_stream_);
        }
        if (!registered_host_fields_.empty() &&
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

    void finalize();

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
    void calibrate_pipeline_pcie();
    bool complete_residency() const noexcept { return complete_residency_; }
    double ram_pcie_fraction() const noexcept { return ram_pcie_fraction_; }
    void finish_pipeline_exchanges() { if(moe_residency_)moe_residency_->apply_pending(); }

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
               << " pipeline_two_stage_serves=" << stats_.pipeline_two_stage_serves
               << " pipeline_route_wait_ns=" << stats_.pipeline_route_wait_ns
               << " pipeline_plan_ns=" << stats_.pipeline_plan_ns
               << " pipeline_fetch_ns=" << stats_.pipeline_fetch_ns
               << " pipeline_cpu_ns=" << stats_.pipeline_cpu_ns
               << " pipeline_gpu_ns=" << stats_.pipeline_gpu_ns
               << " pipeline_gpu_interval_can_include_cpu_wait=" << (stats_.pipeline_two_stage_serves>0)
               << " pipeline_window_serves=" << stats_.pipeline_window_serves
               << " pipeline_window_route_ns=" << stats_.pipeline_window_route_ns
               << " pipeline_window_plan_ns=" << stats_.pipeline_window_plan_ns
               << " pipeline_window_fetch_ns=" << stats_.pipeline_window_fetch_ns
               << " pipeline_window_cpu_ns=" << stats_.pipeline_window_cpu_ns
               << " pipeline_window_gpu_ns=" << stats_.pipeline_window_gpu_ns
               << " pipeline_dma_copies=" << stats_.pipeline_dma_copies
               << " pipeline_window_dma_copies=" << stats_.pipeline_window_dma_copies
               << " pipeline_window_dma_bytes=" << stats_.pipeline_window_dma_bytes
               << " pipeline_ram_registered_bytes=" << pipeline_ram_registered_bytes_
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
               << " pipeline_gate_up_dma_bytes=" << stats_.pipeline_gate_up_dma_bytes
               << " pipeline_down_dma_bytes=" << stats_.pipeline_down_dma_bytes
               << " pipeline_prefetch_experts=" << stats_.pipeline_prefetch_experts
               << " pipeline_prefetch_bytes=" << stats_.pipeline_prefetch_bytes
               << " pipeline_prefetch_hit_bytes=" << stats_.pipeline_prefetch_hit_bytes
               << " pipeline_prefetch_busy_skips=" << stats_.pipeline_prefetch_busy_skips
               << " pipeline_mapped_copy_bytes=" << stats_.pipeline_mapped_copy_bytes
               << " pipeline_mapped_copy_descriptors=" << stats_.pipeline_mapped_copy_descriptors
               << " pipeline_wire_gpu_bytes=" << tensor_nbytes(pipeline_wire_gpu_)
               << " pipeline_sigmoid_table_bytes=" << tensor_nbytes(pipeline_sigmoid_table_)
               << " pipeline_wire_route_bytes=" << pipeline_wire_route_bytes_
               << " pipeline_wire_route_fields=" << pipeline_wire_route_fields_
               << " residency_adapt=" << config_.moe_residency_adapt
               << " residency_warm=" << config_.moe_residency_warm
               << " residency_projection_heat=" << (config_.moe_residency_warm && config_.moe_residency_projection_heat)
               << " residency_observed_routes=" << (moe_residency_?moe_residency_->stats().observed_routes:0)
               << " residency_rounds=" << (moe_residency_?moe_residency_->stats().scheduled_rounds:0)
               << " residency_candidates=" << (moe_residency_?moe_residency_->stats().candidate_bundles:0)
               << " residency_planned_bundles=" << (moe_residency_?moe_residency_->stats().planned_bundles:0)
               << " residency_committed_bundles=" << (moe_residency_?moe_residency_->stats().committed_bundles:0)
               << " residency_memory_rejections=" << (moe_residency_?moe_residency_->stats().memory_rejections:0)
               << " residency_host_pack_bytes=" << (moe_residency_?moe_residency_->stats().host_pack_bytes:0)
               << " residency_direct_upload_bytes=" << (moe_residency_?moe_residency_->stats().direct_upload_bytes:0)
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

private:
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
    bool complete_residency_=false;
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
    friend void prepare_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>&,bool);
    friend void prepare_moe_cpu_budget_comparison(const std::shared_ptr<MoeExpertCache>&,bool,bool);
    friend void prepare_moe_cpu_calibration_comparison(const std::shared_ptr<MoeExpertCache>&,bool);
    friend std::size_t finish_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>&);
    std::unique_ptr<MoeResidencyManager> moe_residency_;
    std::vector<std::array<int,3>> pipeline_bundles_;
    std::unordered_map<std::string,std::unique_ptr<MoeGpuArena>> pipeline_stages_;
    bool pipeline_transfer_cache_=mfq::cuda::runtime_options::transfer_cache();
    bool pipeline_mapped_copy_=mfq::cuda::runtime_options::mapped_copy();
    std::vector<MoeCacheNewLease> pipeline_transfer_slots_;
    mfq_tensor_backend::Tensor pipeline_host_stage_;
    mfq_tensor_backend::Tensor pipeline_wire_gpu_;
    mfq_tensor_backend::Tensor pipeline_prefetch_wire_gpu_,pipeline_prefetch_host_stage_;
    std::vector<mfq::MoeCacheKey> pipeline_prefetched_keys_;
    mfq_tensor_backend::Tensor pipeline_sigmoid_table_;
    int64_t pipeline_wire_route_bytes_=0,pipeline_wire_route_fields_=0;
    std::shared_ptr<mfq_tensor_backend::Tensor> pipeline_ram_complement_;
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
                !prewarming_ && transfer.mapped_source != nullptr;
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
        if (prewarming_) {
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
