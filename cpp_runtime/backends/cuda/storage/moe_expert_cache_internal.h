#pragma once

#include "moe_cache_types_internal.h"

class MoeExpertCache {
public:
    MoeExpertCache(
            int64_t budget_bytes,
            const CudaExecutionConfig& config)
        : budget_bytes_(budget_bytes) {
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
        for (auto & stage : stages_) {
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &stage.done, cudaEventDisableTiming));
        }
        ssd_cache_dir_ = config.moe_ssd_cache_dir;
        // File-backed pages must stay reclaimable, never pin the whole model.
        mapped_gather_enabled_ = config.moe_mapped_gather && ssd_cache_dir_.empty();
        mapped_copy_blocks_ = config.moe_mapped_copy_blocks;
        range_read_pool_ =
            std::make_unique<mfq::cuda::MfeMxfp4ReadPool>(
                config.moe_ssd_io_workers);
        range_overlap_enabled_ = config.moe_ssd_overlap;
        nvq_heterogeneous_enabled_ = config.moe_nvq_heterogeneous;
    }

    ~MoeExpertCache() {
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

    bool file_backed_sources() const noexcept {
        return !ssd_cache_dir_.empty();
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
        const auto route_wait_start = std::chrono::steady_clock::now();
        MFQ_CUDA_CHECK(cudaEventSynchronize(route_done_));
        stats_.route_wait_nanoseconds +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - route_wait_start).count();
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

    void print_stats(std::ostream & stream) const {
        int64_t pinned_stage_bytes = 0, device_stage_bytes = 0;
        for (const auto & stage : stages_) {
            if (stage.host.defined()) pinned_stage_bytes += tensor_nbytes(stage.host);
            if (stage.device.defined()) device_stage_bytes += tensor_nbytes(stage.device);
        }
        stream << "moe_cache_stats"
               << " budget_bytes=" << budget_bytes_
               << " allocated_bytes=" << allocated_bytes_
               << " host_bytes=" << host_bytes_
               << " file_backed_bytes=" << file_backed_bytes_
               << " pinned_stage_bytes=" << pinned_stage_bytes
               << " device_stage_bytes=" << device_stage_bytes
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
               << " staging_ms=" << static_cast<double>(stats_.staging_nanoseconds) / 1.0e6
               << " stage_acquire_ms=" << static_cast<double>(stats_.stage_acquire_nanoseconds) / 1.0e6
               << " route_wait_ms=" << static_cast<double>(stats_.route_wait_nanoseconds) / 1.0e6
               << "\n";
    }

private:
    friend class MoeCachedSource;

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
        auto materialize_source = [this, &range_requests](
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
            } else if (!prewarming_ && transfer.file_backed &&
                    transfer.packed_weight && range_read_pool_->workers() > 1) {
                range_requests.push_back({
                    nullptr, nullptr,
                    std::span<uint8_t>(destination, static_cast<size_t>(transfer.nbytes)),
                    transfer.source,
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
        const auto stage_acquire_start = std::chrono::steady_clock::now();
        auto & stage = acquire_stage(total_bytes, true);
        stats_.stage_acquire_nanoseconds +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - stage_acquire_start).count();
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
        const auto staging_start = std::chrono::steady_clock::now();
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
        stats_.staging_nanoseconds +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - staging_start).count();
        if (scatter_descriptor != staged_count ||
                mapped_descriptor != mapped_count) {
            throw std::runtime_error(
                "MoE cache transfer descriptor count mismatch");
        }
        if (waits_for_compute && compute_done_recorded_) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                weight_stream_, compute_done_, 0));
        }
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            stage.device.data_ptr<uint8_t>(),
            staging,
            static_cast<size_t>(total_bytes),
            cudaMemcpyHostToDevice,
            weight_stream_));
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

    void invalidate(const mfq::MoeCacheKey & key, int slot);

    int64_t budget_bytes_ = 0;
    int64_t allocated_bytes_ = 0;
    int64_t host_bytes_ = 0;
    int64_t file_backed_bytes_ = 0;
    std::string ssd_cache_dir_;
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
    bool nvq_heterogeneous_enabled_ = true;
};
