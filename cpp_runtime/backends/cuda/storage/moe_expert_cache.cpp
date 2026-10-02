#include "moe_cached_source_internal.h"

std::shared_ptr<MoeCachedSource> MoeExpertCache::register_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> cpu,
        int minimum_slots,
        int layer_id,
        std::string projection_role) {
    const int id = static_cast<int>(sources_.size());
    auto source = std::make_shared<MoeCachedSource>(
        this, id, name, std::move(cpu), minimum_slots,
        layer_id, std::move(projection_role), nullptr);
    host_bytes_ += source->host_bytes();
    sources_.push_back(source);
    return source;
}

std::shared_ptr<MoeCachedSource> MoeExpertCache::register_range_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> metadata,
        std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> store,
        int minimum_slots,
        int layer_id,
        std::string projection_role) {
    const int id = static_cast<int>(sources_.size());
    auto source = std::make_shared<MoeCachedSource>(
        this, id, name, std::move(metadata), minimum_slots,
        layer_id, std::move(projection_role), std::move(store));
    host_bytes_ += source->host_bytes();
    sources_.push_back(source);
    return source;
}

void MoeExpertCache::finalize() {
    if (finalized_) return;
    if (sources_.empty()) {
        throw std::runtime_error(
            "MoE cache has no registered expert sources");
    }
    std::vector<mfq::MoeArenaDemand> demands;
    demands.reserve(arenas_.size());
    for (const auto & item : arenas_) {
        const auto & arena = *item.second;
        demands.push_back({
            arena.signature,
            arena.slot_bytes,
            arena.minimum_slots,
            arena.registered_experts,
        });
    }

    if (profile_.has_value()) {
        std::unordered_map<
            int, std::vector<std::shared_ptr<MoeCachedSource>>>
            layer_sources;
        std::unordered_map<int, int> layer_experts;
        std::unordered_map<int, std::unordered_set<std::string>>
            layer_roles;
        for (const auto & source : sources_) {
            if (source->layer_id() < 0) continue;
            if (source->projection_role().empty()) {
                throw std::runtime_error(
                    "profiled MoE cache source has no projection role");
            }
            if (!layer_roles[source->layer_id()]
                     .insert(source->projection_role()).second) {
                throw std::runtime_error(
                    "profiled MoE cache layer has a duplicate projection role");
            }
            const auto inserted = layer_experts.emplace(
                source->layer_id(), source->n_experts());
            if (!inserted.second &&
                    inserted.first->second != source->n_experts()) {
                throw std::runtime_error(
                    "profiled MoE cache layer sources disagree on expert count");
            }
            layer_sources[source->layer_id()].push_back(source);
        }
        mfq::validate_moe_cache_profile(
            *profile_, layer_experts);

        std::unordered_map<std::string, int> required_slots;
        std::unordered_map<std::string, int> selected_slots;
        int64_t required_bytes = 0;
        for (const auto & demand : demands) {
            required_slots[demand.signature] =
                demand.minimum_slots;
            required_bytes +=
                static_cast<int64_t>(demand.minimum_slots) *
                demand.slot_bytes;
        }
        if (required_bytes > budget_bytes_) {
            throw std::runtime_error(
                "MoE cache budget is below the minimum decode working set");
        }

        auto bundle_bytes = [&](int layer, int expert) {
            int64_t bytes = 0;
            for (const auto & source : layer_sources.at(layer)) {
                const int64_t source_bytes =
                    source->bytes_for_expert(expert);
                if (source_bytes >
                        std::numeric_limits<int64_t>::max() - bytes) {
                    throw std::overflow_error(
                        "MoE cache prewarm bundle byte count overflows int64");
                }
                bytes += source_bytes;
            }
            return bytes;
        };

        auto select_candidate = [&](
                const mfq::MoeProfileCandidate & candidate,
                int64_t layer_limit,
                std::unordered_map<int, int64_t> * layer_used) {
            const auto source_it =
                layer_sources.find(candidate.layer);
            if (source_it == layer_sources.end() ||
                    source_it->second.empty()) {
                return false;
            }
            std::unordered_map<std::string, int> needed;
            for (const auto & source : source_it->second) {
                ++needed[
                    source->arena_for_expert(
                        candidate.expert)->signature];
            }
            int64_t incremental_bytes = 0;
            for (const auto & item : needed) {
                const auto arena_it = arenas_.find(item.first);
                if (arena_it == arenas_.end()) {
                    throw std::runtime_error(
                        "MoE cache profile references an unknown arena");
                }
                const auto & arena = *arena_it->second;
                const int selected =
                    selected_slots[item.first] + item.second;
                if (selected > arena.registered_experts) {
                    return false;
                }
                const int new_required = std::max(
                    required_slots[item.first], selected);
                incremental_bytes +=
                    static_cast<int64_t>(
                        new_required -
                        required_slots[item.first]) *
                    arena.slot_bytes;
            }
            const int64_t bytes =
                bundle_bytes(candidate.layer, candidate.expert);
            if (layer_used != nullptr &&
                    bytes >
                        layer_limit -
                        (*layer_used)[candidate.layer]) {
                return false;
            }
            if (incremental_bytes >
                    budget_bytes_ - required_bytes) {
                return false;
            }
            for (const auto & item : needed) {
                selected_slots[item.first] += item.second;
                required_slots[item.first] = std::max(
                    required_slots[item.first],
                    selected_slots[item.first]);
            }
            required_bytes += incremental_bytes;
            if (layer_used != nullptr) {
                (*layer_used)[candidate.layer] += bytes;
            }
            prewarm_selected_.push_back(candidate);
            return true;
        };

        const auto candidates =
            mfq::order_moe_profile_candidates(
                *profile_, false);
        int64_t frequency_bytes = 0;
        std::unordered_set<int> ranking_layers;
        for (const auto & candidate : candidates) {
            if (candidate.has_frequency) {
                if (select_candidate(
                        candidate,
                        std::numeric_limits<int64_t>::max(),
                        nullptr)) {
                    frequency_bytes +=
                        bundle_bytes(
                            candidate.layer,
                            candidate.expert);
                }
            } else {
                ranking_layers.insert(candidate.layer);
            }
        }

        std::unordered_map<int, int64_t> rank_layer_total;
        int64_t rank_model_total = 0;
        for (int layer : ranking_layers) {
            int64_t total = 0;
            for (int expert = 0;
                 expert < layer_experts.at(layer);
                 ++expert) {
                total += bundle_bytes(layer, expert);
            }
            rank_layer_total[layer] = total;
            rank_model_total += total;
        }
        const int64_t rank_budget =
            std::max<int64_t>(
                0, budget_bytes_ - frequency_bytes);
        std::unordered_map<int, int64_t> rank_layer_limit;
        for (const auto & item : rank_layer_total) {
            const long double fraction =
                rank_model_total > 0
                ? static_cast<long double>(item.second) /
                    static_cast<long double>(rank_model_total)
                : 0.0L;
            rank_layer_limit[item.first] =
                static_cast<int64_t>(
                    static_cast<long double>(rank_budget) *
                    fraction);
        }
        std::unordered_map<int, int64_t> rank_layer_used;
        for (const auto & candidate : candidates) {
            if (candidate.has_frequency) continue;
            (void)select_candidate(
                candidate,
                rank_layer_limit.at(candidate.layer),
                &rank_layer_used);
        }

        for (auto & demand : demands) {
            demand.minimum_slots = std::max(
                demand.minimum_slots,
                required_slots.at(demand.signature));
        }
    }
    const auto plan =
        mfq::plan_moe_arena_slots(budget_bytes_, demands);
    for (auto & item : arenas_) {
        auto & arena = *item.second;
        arena.slots = plan.at(arena.signature);
        arena.fields.reserve(arena.layouts.size());
        for (size_t index = 0;
             index < arena.layouts.size();
             ++index) {
            const auto & layout = arena.layouts[index];
            if (layout.slot_shape.empty()) {
                throw std::runtime_error(
                    "MoE cache field has no expert-major leading dimension");
            }
            auto shape = layout.slot_shape;
            shape[0] *= arena.slots;
            arena.fields.push_back(mfq_tensor_backend::empty(
                shape,
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCUDA)
                    .dtype(layout.scalar_type)));
        }
        arena.book =
            std::make_unique<mfq::MoeCacheSlotBook>(
                arena.slots);
        allocated_bytes_ +=
            static_cast<int64_t>(arena.slots) *
            arena.slot_bytes;
        std::cerr
            << "moe_cache_arena"
            << " signature=" << arena.signature
            << " slots=" << arena.slots
            << " slot_bytes=" << arena.slot_bytes
            << " bytes="
            << static_cast<int64_t>(arena.slots) *
                arena.slot_bytes
            << std::endl;
    }
    if (allocated_bytes_ > budget_bytes_) {
        throw std::runtime_error(
            "MoE cache arena allocation exceeded its budget");
    }
    for (auto & source : sources_) source->finalize();
    finalized_ = true;
    std::cerr
        << "moe_cache_ready"
        << " sources=" << sources_.size()
        << " host_bytes=" << host_bytes_
        << " budget_bytes=" << budget_bytes_
        << " allocated_bytes=" << allocated_bytes_
        << " mapped_gather=" << (mapped_gather_enabled_ ? 1 : 0)
        << " mapped_registered_bytes=" << mapped_registered_bytes_
        << " mapped_copy_blocks=" << mapped_copy_blocks_
        << std::endl;
    if (profile_.has_value()) prewarm();
}

void MoeExpertCache::prewarm() {
    if (!finalized_ || !profile_.has_value()) return;

    std::unordered_map<
        int, std::vector<std::shared_ptr<MoeCachedSource>>>
        layer_sources;
    for (const auto & source : sources_) {
        if (source->layer_id() < 0) continue;
        layer_sources[source->layer_id()].push_back(source);
    }

    const auto started = std::chrono::steady_clock::now();
    std::unordered_map<
        MoeCachedSource *, std::vector<int32_t>>
        source_experts;
    for (auto selected_it = prewarm_selected_.rbegin();
         selected_it != prewarm_selected_.rend();
         ++selected_it) {
        for (const auto & source :
             layer_sources.at(selected_it->layer)) {
            source_experts[source.get()].push_back(
                selected_it->expert);
        }
    }
    prewarming_ = true;
    try {
        for (const auto & source : sources_) {
            const auto found = source_experts.find(source.get());
            if (found == source_experts.end()) continue;
            prepare(*source, found->second, true);
        }
        if (transfer_ready_recorded_) {
            MFQ_CUDA_CHECK(
                cudaEventSynchronize(transfer_ready_));
        }
        prewarming_ = false;
    } catch (...) {
        prewarming_ = false;
        throw;
    }
    for (auto selected_it = prewarm_selected_.rbegin();
         selected_it != prewarm_selected_.rend();
         ++selected_it) {
        for (const auto & source :
             layer_sources.at(selected_it->layer)) {
            source->touch_expert(
                selected_it->expert);
        }
    }
    const auto stopped = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            stopped - started).count();
    const int64_t prewarm_h2d_bytes = stats_.h2d_bytes;
    const int64_t prewarm_range_read_bytes = stats_.range_read_bytes;
    const int64_t prewarm_range_read_calls = stats_.range_read_calls;
    const int64_t prewarm_range_file_opens = stats_.range_file_opens;
    const double prewarm_range_read_ms =
        static_cast<double>(stats_.range_read_nanoseconds) / 1.0e6;
    const int64_t projection_entries =
        stats_.prefetch_misses;
    const int64_t unfilled_bytes =
        std::max<int64_t>(
            0, allocated_bytes_ - prewarm_h2d_bytes);
    std::cout
        << "moe_cache_prewarm"
        << " profile=" << profile_->path
        << " expert_bundles=" << prewarm_selected_.size()
        << " projection_entries=" << projection_entries
        << " h2d_bytes=" << prewarm_h2d_bytes
        << " range_read_bytes=" << prewarm_range_read_bytes
        << " range_read_calls=" << prewarm_range_read_calls
        << " range_file_opens=" << prewarm_range_file_opens
        << " range_read_ms=" << prewarm_range_read_ms
        << " time_ms=" << elapsed_ms
        << " unfilled_bytes=" << unfilled_bytes
        << "\n";
    stats_ = {};
}

void MoeExpertCache::invalidate(
        const mfq::MoeCacheKey & key,
        int slot) {
    if (key.source < 0 ||
        key.source >= static_cast<int>(sources_.size())) {
        throw std::runtime_error(
            "MoE cache eviction references an invalid source");
    }
    sources_.at(static_cast<size_t>(key.source))
        ->invalidate(key.cohort, key.expert, slot);
}

void MoeExpertCache::append_source_transfers(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch,
        std::vector<MoeCacheTransfer> & transfers,
        bool & replaced_occupied,
        std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> * held_slots,
        std::vector<MoeCacheNewLease> * new_leases) {
    for (int expert : experts) {
        const int cohort_index =
            source.expert_to_cohort_.at(
                static_cast<size_t>(expert));
        const int local_index =
            source.expert_to_local_.at(
                static_cast<size_t>(expert));
        auto & cohort =
            source.cohorts_.at(
                static_cast<size_t>(cohort_index));
        auto & arena = *cohort.arena;
        const mfq::MoeCacheKey key{
            source.id_, cohort_index, expert};
        auto lease = arena.book->acquire(key);
        if (held_slots != nullptr &&
                !arena.book->inflight(lease.slot)) {
            arena.book->mark_inflight(lease.slot);
            held_slots->emplace_back(arena.book.get(), lease.slot);
        }
        if (lease.hit) {
            if (prefetch) {
                ++stats_.prefetch_hits;
            } else {
                ++stats_.demand_hits;
            }
        } else {
            if (new_leases != nullptr) {
                new_leases->push_back({
                    arena.book.get(),
                    key,
                    lease.slot,
                    lease.generation,
                });
            }
            if (prefetch) {
                ++stats_.prefetch_misses;
            } else {
                ++stats_.demand_misses;
            }
            if (lease.replaced.has_value()) {
                ++stats_.evictions;
                replaced_occupied = true;
                invalidate(*lease.replaced, lease.slot);
            }
            for (size_t field = 0;
                 field < cohort.bytes_per_expert.size();
                 ++field) {
                const int64_t nbytes =
                    cohort.bytes_per_expert[field];
                if (nbytes == 0) continue;
                auto & gpu_field =
                    arena.fields[field];
                if (cohort.range_store) {
                    const auto & part =
                        cohort.range_store->part(expert, field);
                    transfers.push_back({
                        nullptr,
                        reinterpret_cast<uint8_t *>(
                            gpu_field.data_ptr()) +
                            static_cast<int64_t>(lease.slot) * nbytes,
                        nbytes,
                        true,
                        nullptr,
                        cohort.range_store.get(),
                        &part,
                    });
                    continue;
                }
                const auto & cpu_field =
                    cohort.cpu_fields[field];
                transfers.push_back({
                    reinterpret_cast<const uint8_t *>(
                        cpu_field.data_ptr()) +
                        static_cast<int64_t>(local_index) *
                        nbytes,
                    reinterpret_cast<uint8_t *>(
                        gpu_field.data_ptr()) +
                        static_cast<int64_t>(lease.slot) *
                        nbytes,
                    nbytes,
                    true,
                    cohort.mapped_fields.at(field) == nullptr
                        ? nullptr
                        : cohort.mapped_fields.at(field) +
                            static_cast<int64_t>(local_index) * nbytes,
                });
            }
        }
        if (cohort.host_map[
                static_cast<size_t>(expert)] != lease.slot) {
            cohort.host_map[
                static_cast<size_t>(expert)] = lease.slot;
            cohort.map_dirty = true;
        }
    }

    for (auto & cohort : source.cohorts_) {
        if (!cohort.map_dirty) continue;
        auto & active =
            source.active_->pools.at(
                static_cast<size_t>(cohort.index));
        transfers.push_back({
            reinterpret_cast<const uint8_t *>(
                cohort.host_map.data()),
            reinterpret_cast<uint8_t *>(
                active.expert_local.data_ptr<int32_t>()),
            static_cast<int64_t>(
                cohort.host_map.size() *
                sizeof(int32_t)),
            false,
        });
        cohort.map_dirty = false;
    }
}

void MoeExpertCache::rollback_preparation(
        const std::vector<MoeCacheNewLease> & new_leases,
        const std::vector<
            std::pair<mfq::MoeCacheSlotBook *, int>> & held_slots) noexcept {
    (void)cudaStreamSynchronize(weight_stream_);
    for (auto lease = new_leases.rbegin();
         lease != new_leases.rend();
         ++lease) {
        try {
            invalidate(lease->key, lease->slot);
            (void)lease->book->discard(
                lease->key, lease->slot, lease->generation);
        } catch (...) {
        }
    }
    for (const auto & held : held_slots) {
        try {
            held.first->clear_inflight(held.second);
        } catch (...) {
        }
    }
}

bool MoeExpertCache::begin_deferred_range_read(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts) {
    if (!range_overlap_enabled_ || prewarming_ || !source.range_store_ ||
            experts.empty()) {
        return false;
    }
    finish_deferred_range_read(source);

    std::unordered_map<
        MoeGpuArena *,
        std::unordered_set<mfq::MoeCacheKey, mfq::MoeCacheKeyHash>>
        arena_demands;
    for (int expert : experts) {
        if (expert < 0 || expert >= source.n_experts()) return false;
        const int cohort_index = source.expert_to_cohort_.at(
            static_cast<size_t>(expert));
        auto & cohort = source.cohorts_.at(
            static_cast<size_t>(cohort_index));
        arena_demands[cohort.arena].insert(
            {source.id_, cohort_index, expert});
    }
    for (const auto & item : arena_demands) {
        if (item.second.size() >
                static_cast<size_t>(item.first->book->capacity())) {
            return false;
        }
    }

    auto pending = std::make_unique<MoePendingRangeRead>();
    for (const auto & item : arena_demands) {
        auto * book = item.first->book.get();
        for (const auto & key : item.second) {
            const int slot = book->slot_for(key);
            if (slot >= 0 && !book->inflight(slot)) {
                book->mark_inflight(slot);
                pending->held_slots.emplace_back(book, slot);
            }
        }
    }

    try {
        append_source_transfers(
            source,
            experts,
            true,
            pending->transfers,
            pending->replaced_occupied,
            &pending->held_slots,
            &pending->new_leases);

        int64_t range_bytes = 0;
        size_t range_count = 0;
        for (const auto & transfer : pending->transfers) {
            if (transfer.range_store == nullptr) continue;
            range_bytes = (range_bytes + 15) & ~int64_t{15};
            if (transfer.nbytes >
                    std::numeric_limits<int64_t>::max() - range_bytes) {
                throw std::overflow_error(
                    "deferred MoE range byte count overflows int64");
            }
            range_bytes += transfer.nbytes;
            ++range_count;
        }
        if (range_count == 0) {
            submit_transfers(
                pending->transfers,
                pending->replaced_occupied,
                false);
            for (const auto & held : pending->held_slots) {
                held.first->clear_inflight(held.second);
            }
            return true;
        }

        pending->host = mfq_tensor_backend::empty(
            {range_bytes},
            mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCPU)
                .dtype(mfq_tensor_backend::kUInt8)
                .pinned_memory(true));
        auto * staging = pending->host.data_ptr<uint8_t>();
        std::vector<mfq::cuda::MfeMxfp4ReadRequest> requests;
        requests.reserve(range_count);
        int64_t offset = 0;
        for (auto & transfer : pending->transfers) {
            if (transfer.range_store == nullptr) continue;
            offset = (offset + 15) & ~int64_t{15};
            requests.push_back({
                transfer.range_store,
                transfer.range_part,
                std::span<uint8_t>(
                    staging + offset,
                    static_cast<size_t>(transfer.nbytes)),
            });
            transfer.source = staging + offset;
            transfer.range_store = nullptr;
            transfer.range_part = nullptr;
            offset += transfer.nbytes;
        }
        pending->ticket = range_read_pool_->submit(requests);
        const auto inserted = pending_range_reads_.try_emplace(
            source.id_, std::move(pending));
        if (!inserted.second) {
            throw std::runtime_error(
                "MoE source already has a deferred range read");
        }
        ++stats_.range_overlap_batches;
        return true;
    } catch (...) {
        if (pending) {
            rollback_preparation(
                pending->new_leases, pending->held_slots);
        }
        throw;
    }
}

void MoeExpertCache::finish_deferred_range_read(
        MoeCachedSource & source) {
    const auto found = pending_range_reads_.find(source.id_);
    if (found == pending_range_reads_.end()) return;
    auto pending = std::move(found->second);
    pending_range_reads_.erase(found);
    try {
        const auto wait_begin = std::chrono::steady_clock::now();
        const auto range_stats = pending->ticket.wait();
        stats_.range_overlap_wait_nanoseconds +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wait_begin).count();
        record_range_read(range_stats);
        submit_transfers(
            pending->transfers,
            pending->replaced_occupied,
            true);
    } catch (...) {
        rollback_preparation(
            pending->new_leases, pending->held_slots);
        throw;
    }
    for (const auto & held : pending->held_slots) {
        held.first->clear_inflight(held.second);
    }
}

bool MoeExpertCache::prepare(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch) {
    if (!finalized_) {
        throw std::runtime_error(
            "MoE cache must be finalized before inference");
    }
    finish_deferred_range_read(source);
    if (experts.empty()) return false;

    std::unordered_map<
        MoeGpuArena *,
        std::unordered_set<mfq::MoeCacheKey, mfq::MoeCacheKeyHash>>
        arena_demands;
    for (int expert : experts) {
        if (expert < 0 || expert >= source.n_experts()) return false;
        const int cohort_index = source.expert_to_cohort_.at(
            static_cast<size_t>(expert));
        auto & cohort = source.cohorts_.at(
            static_cast<size_t>(cohort_index));
        arena_demands[cohort.arena].insert(
            {source.id_, cohort_index, expert});
    }
    for (const auto & item : arena_demands) {
        if (item.second.size() >
                static_cast<size_t>(item.first->book->capacity())) {
            return false;
        }
    }

    std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> held_slots;
    for (const auto & item : arena_demands) {
        auto * book = item.first->book.get();
        for (const auto & key : item.second) {
            const int slot = book->slot_for(key);
            if (slot >= 0 && !book->inflight(slot)) {
                book->mark_inflight(slot);
                held_slots.emplace_back(book, slot);
            }
        }
    }
    std::vector<MoeCacheTransfer> transfers;
    std::vector<MoeCacheNewLease> new_leases;
    bool replaced_occupied = false;
    try {
        append_source_transfers(
            source, experts, prefetch, transfers,
            replaced_occupied, &held_slots, &new_leases);
        submit_transfers(
            transfers,
            replaced_occupied,
            !prefetch);
    } catch (...) {
        rollback_preparation(new_leases, held_slots);
        throw;
    }
    for (const auto & held : held_slots) {
        held.first->clear_inflight(held.second);
    }
    return true;
}

bool MoeExpertCache::prepare_bundle(
        const std::vector<MoeCachedSource *> & sources,
        const std::vector<int32_t> & experts) {
    if (!finalized_) {
        throw std::runtime_error(
            "MoE cache must be finalized before inference");
    }
    for (auto * source : sources) {
        if (source != nullptr && source->cache_ == this) {
            finish_deferred_range_read(*source);
        }
    }
    if (sources.empty() || experts.empty()) return false;

    std::unordered_map<
        MoeGpuArena *,
        std::unordered_set<mfq::MoeCacheKey, mfq::MoeCacheKeyHash>>
        arena_demands;
    for (auto * source : sources) {
        if (source == nullptr || source->cache_ != this) return false;
        for (int expert : experts) {
            if (expert < 0 || expert >= source->n_experts()) return false;
            const int cohort_index = source->expert_to_cohort_.at(
                static_cast<size_t>(expert));
            auto & cohort = source->cohorts_.at(
                static_cast<size_t>(cohort_index));
            arena_demands[cohort.arena].insert(
                {source->id_, cohort_index, expert});
        }
    }
    for (const auto & item : arena_demands) {
        if (item.second.size() >
                static_cast<size_t>(item.first->book->capacity())) {
            return false;
        }
    }

    std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> held_slots;
    for (const auto & item : arena_demands) {
        auto * book = item.first->book.get();
        for (const auto & key : item.second) {
            const int slot = book->slot_for(key);
            if (slot >= 0 && !book->inflight(slot)) {
                book->mark_inflight(slot);
                held_slots.emplace_back(book, slot);
            }
        }
    }
    std::vector<MoeCacheTransfer> transfers;
    std::vector<MoeCacheNewLease> new_leases;
    bool replaced_occupied = false;
    try {
        for (auto * source : sources) {
            append_source_transfers(
                *source, experts, true, transfers,
                replaced_occupied, &held_slots, &new_leases);
        }
        submit_transfers(
            transfers, replaced_occupied, false);
    } catch (...) {
        rollback_preparation(new_leases, held_slots);
        throw;
    }
    for (const auto & held : held_slots) {
        held.first->clear_inflight(held.second);
    }
    return true;
}

bool MoeExpertCache::prepare_bundle_deferred(
        const std::vector<MoeCachedSource *> & ready_sources,
        MoeCachedSource & deferred_source,
        const std::vector<int32_t> & experts) {
    if (!range_overlap_enabled_ || !deferred_source.range_store_) {
        auto sources = ready_sources;
        sources.push_back(&deferred_source);
        return prepare_bundle(sources, experts);
    }
    if (!prepare_bundle(ready_sources, experts)) return false;
    return begin_deferred_range_read(deferred_source, experts);
}

static MfeWeight wrap_cached_moe_source(
        const std::shared_ptr<MoeCachedSource> & source,
        const std::shared_ptr<MixedMoeRuntime> & cpu) {
    MfeWeight result =
        cpu_mixed_moe_metadata(cpu);
    result.mixed_weight_bytes = source->logical_weight_bytes();
    result.cached_source = source;
    result.cache_prefetch = [source](
            const MoeRoutePlan & route) {
        source->prefetch(route);
    };
    result.cache_prefetch_begin = [source](
            const MoeRoutePlan & route) {
        source->begin_prefetch(route);
    };
    result.mixed_forward = [source](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        return source->forward(execution, x, route);
    };
    result.mixed_prequantized_forward = [source](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        return source->forward_prequantized(execution, x, route);
    };
    result.mixed_glu_output_forward = [source](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) {
        return source->forward_glu_output(
            execution, x, route, gelu);
    };
    result.mixed_glu_forward = [source](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            bool gelu) {
        return source->forward_glu(
            execution, gate_up, route, gelu);
    };
    if (source->supports_clamped_swiglu()) {
        result.mixed_clamped_swiglu_forward = [source](
                CudaExecutionContext& execution,
                mfq_tensor_backend::Tensor gate_up,
                const MoeRoutePlan & route,
                double limit) {
            return source->forward_clamped_swiglu(
                execution, gate_up, route, limit);
        };
    }
    return result;
}

bool prefetch_cached_moe_projection_bundle(
        const MfeWeight & gate,
        const MfeWeight & up,
        const MfeWeight & down,
        const MoeRoutePlan & route) {
    if (!gate.cached_source ||
            !up.cached_source ||
            !down.cached_source) {
        return false;
    }
    return MoeCachedSource::prefetch_bundle(
        {gate.cached_source, up.cached_source, down.cached_source},
        route);
}

bool prefetch_cached_moe_projection_bundle(
        const MfeWeight & gate_up,
        const MfeWeight & down,
        const MoeRoutePlan & route) {
    if (!gate_up.cached_source || !down.cached_source) {
        return false;
    }
    return MoeCachedSource::prefetch_bundle(
        {gate_up.cached_source, down.cached_source}, route);
}

MfeWeight cache_moe_weight(
        const std::shared_ptr<MoeExpertCache>& cache,
        const std::string& name,
        const std::shared_ptr<MixedMoeRuntime>& runtime,
        int minimum_slots,
        int layer_id,
        const std::string& projection_role,
        std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> range_store) {
    auto source = range_store
        ? cache->register_range_source(name, runtime, std::move(range_store),
                                       minimum_slots, layer_id, projection_role)
        : cache->register_source(name, runtime, minimum_slots, layer_id, projection_role);
    // The cache owns registered sources. Export an alias that retains the cache,
    // so weights and copied forward/prefetch operations cannot outlive its arenas.
    return wrap_cached_moe_source(
        std::shared_ptr<MoeCachedSource>(cache, source.get()), runtime);
}

std::shared_ptr<MoeExpertCache> make_moe_expert_cache(
        std::int64_t bytes,
        const CudaExecutionConfig& config) {
    return std::make_shared<MoeExpertCache>(bytes, config);
}

bool moe_expert_cache_has_sources(
        const std::shared_ptr<MoeExpertCache>& cache) {
    return cache && cache->has_sources();
}

bool moe_expert_cache_finalized(
        const std::shared_ptr<MoeExpertCache>& cache) {
    return cache && cache->finalized();
}

void finalize_moe_expert_cache(
        const std::shared_ptr<MoeExpertCache>& cache) {
    if (cache && !cache->finalized()) cache->finalize();
}

void print_moe_expert_cache_stats(
        const std::shared_ptr<MoeExpertCache>& cache,
        std::ostream& output) {
    if (cache) cache->print_stats(output);
}

void set_moe_expert_cache_profile(
        MoeExpertCache& cache,
        mfq::MoeCacheProfile profile) {
    cache.set_profile(std::move(profile));
}
