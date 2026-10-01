#pragma once

#include "moe_expert_cache_internal.h"

struct MoeCachedCohort {
    int index = -1;
    const MixedMoePool * cpu = nullptr;
    MoeGpuArena * arena = nullptr;
    std::vector<mfq_tensor_backend::Tensor> cpu_fields;
    std::vector<const uint8_t *> mapped_fields;
    std::vector<int64_t> bytes_per_expert;
    std::vector<int32_t> expert_to_local;
    std::vector<int32_t> host_map;
    std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> range_store;
    bool map_dirty = false;
    MixedMoePool active;
};

class MoeCachedSource : public std::enable_shared_from_this<MoeCachedSource> {
public:
    MoeCachedSource(
            MoeExpertCache * cache,
            int id,
            std::string name,
            std::shared_ptr<MixedMoeRuntime> cpu,
            int minimum_slots,
            int layer_id,
            std::string projection_role,
            std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> range_store)
        : cache_(cache),
          id_(id),
          name_(std::move(name)),
          layer_id_(layer_id),
          projection_role_(std::move(projection_role)),
          cpu_(std::move(cpu)),
          range_store_(std::move(range_store)),
          expert_to_cohort_(
              static_cast<size_t>(cpu_->n_experts), -1),
          expert_to_local_(
              static_cast<size_t>(cpu_->n_experts), -1) {
        if (minimum_slots <= 0) {
            throw std::runtime_error(
                "MoE cache source minimum slots must be positive");
        }
        cohorts_.reserve(cpu_->pools.size());
        if (range_store_) {
            if (cpu_->pools.size() != 1 ||
                    cpu_->pools.front().family != MixedMoeFamily::Mxfp4 ||
                    cpu_->n_experts != range_store_->num_experts() ||
                    cpu_->out_per_expert != range_store_->out_per_expert() ||
                    cpu_->neuron_len != range_store_->neuron_len() ||
                    range_store_->values_bytes_per_expert() >
                        static_cast<uint64_t>(
                            std::numeric_limits<int64_t>::max()) ||
                    range_store_->scales_bytes_per_expert() >
                        static_cast<uint64_t>(
                            std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error(
                    "invalid exact-range MXFP4 cache metadata");
            }
            const auto & pool = cpu_->pools.front();
            MoeCachedCohort cohort;
            cohort.index = 0;
            cohort.cpu = &pool;
            cohort.range_store = range_store_;
            const int64_t values = static_cast<int64_t>(
                range_store_->values_bytes_per_expert());
            const int64_t scales = static_cast<int64_t>(
                range_store_->scales_bytes_per_expert());
            cohort.arena = cache_->register_cohort_layout(
                pool,
                cpu_->out_per_expert,
                cpu_->neuron_len,
                minimum_slots,
                cpu_->n_experts,
                {
                    {
                        mfq_tensor_backend::kUInt8,
                        {cpu_->out_per_expert, cpu_->neuron_len / 2},
                        values,
                        1,
                    },
                    {
                        mfq_tensor_backend::kUInt8,
                        {cpu_->out_per_expert, cpu_->neuron_len / 32},
                        scales,
                        1,
                    },
                });
            cohort.bytes_per_expert = {values, scales};
            cohort.mapped_fields = {nullptr, nullptr};
            cohort.expert_to_local.resize(
                static_cast<size_t>(cpu_->n_experts));
            cohort.host_map.assign(
                static_cast<size_t>(cpu_->n_experts), -1);
            for (int expert = 0; expert < cpu_->n_experts; ++expert) {
                cohort.expert_to_local[static_cast<size_t>(expert)] = expert;
                expert_to_cohort_[static_cast<size_t>(expert)] = 0;
                expert_to_local_[static_cast<size_t>(expert)] = expert;
            }
            cohorts_.push_back(std::move(cohort));
        } else {
            for (int cohort_index = 0;
                 cohort_index < static_cast<int>(cpu_->pools.size());
                 ++cohort_index) {
                const auto & pool =
                    cpu_->pools.at(static_cast<size_t>(cohort_index));
                MoeCachedCohort cohort;
                cohort.index = cohort_index;
                cohort.cpu = &pool;
                cohort.arena = cache_->register_cohort(
                    pool, cpu_->out_per_expert, cpu_->neuron_len,
                    minimum_slots);
                cohort.cpu_fields = moe_cache_fields(pool);
                cohort.expert_to_local.assign(
                    static_cast<size_t>(cpu_->n_experts), -1);
                cohort.host_map.assign(
                    static_cast<size_t>(cpu_->n_experts), -1);
                const auto * local =
                    pool.expert_local.data_ptr<int32_t>();
                for (int expert = 0; expert < cpu_->n_experts; ++expert) {
                    const int local_index = local[expert];
                    cohort.expert_to_local[
                        static_cast<size_t>(expert)] = local_index;
                    if (local_index < 0) continue;
                    if (expert_to_cohort_[
                            static_cast<size_t>(expert)] >= 0) {
                        throw std::runtime_error(
                            "MoE cache source has duplicate expert ownership");
                    }
                    expert_to_cohort_[
                        static_cast<size_t>(expert)] = cohort_index;
                    expert_to_local_[
                        static_cast<size_t>(expert)] = local_index;
                }
                for (const auto & field : cohort.cpu_fields) {
                    const int64_t nbytes = tensor_nbytes(field);
                    if (nbytes % pool.local_experts != 0) {
                        throw std::runtime_error(
                            "MoE cache field byte count is not expert aligned");
                    }
                    cohort.bytes_per_expert.push_back(
                        nbytes / pool.local_experts);
                    cohort.mapped_fields.push_back(
                        cache_->register_mapped_field(field));
                }
                cohorts_.push_back(std::move(cohort));
            }
        }
        if (std::any_of(
                expert_to_cohort_.begin(),
                expert_to_cohort_.end(),
                [](int value) { return value < 0; })) {
            throw std::runtime_error(
                "MoE cache source does not cover every expert");
        }
    }

    int id() const noexcept {
        return id_;
    }

    const std::string & name() const noexcept {
        return name_;
    }

    int layer_id() const noexcept {
        return layer_id_;
    }

    const std::string & projection_role() const noexcept {
        return projection_role_;
    }

    int n_experts() const noexcept {
        return cpu_->n_experts;
    }

    MoeGpuArena * arena_for_expert(int expert) const {
        if (expert < 0 || expert >= cpu_->n_experts) {
            throw std::out_of_range(
                "MoE cache profile expert is out of range");
        }
        const int cohort = expert_to_cohort_.at(
            static_cast<size_t>(expert));
        return cohorts_.at(static_cast<size_t>(cohort)).arena;
    }

    int64_t bytes_for_expert(int expert) const {
        return arena_for_expert(expert)->slot_bytes;
    }

    void touch_expert(int expert) {
        const int cohort = expert_to_cohort_.at(
            static_cast<size_t>(expert));
        auto & arena = *cohorts_.at(
            static_cast<size_t>(cohort)).arena;
        const int slot = arena.book->slot_for(
            {id_, cohort, expert});
        if (slot < 0) {
            throw std::runtime_error(
                "prewarmed MoE expert is absent from its arena");
        }
        arena.book->touch(slot);
    }

    int64_t host_bytes() const {
        return mixed_moe_storage_bytes(*cpu_);
    }

    int64_t logical_weight_bytes() const {
        if (!range_store_) return host_bytes();
        return range_store_->record().nbytes >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
            ? std::numeric_limits<int64_t>::max()
            : static_cast<int64_t>(range_store_->record().nbytes);
    }

    void finalize() {
        if (active_) return;
        active_ = std::make_shared<MixedMoeRuntime>();
        active_->n_experts = cpu_->n_experts;
        active_->out_per_expert = cpu_->out_per_expert;
        active_->neuron_len = cpu_->neuron_len;
        active_->pools.reserve(cohorts_.size());
        for (auto & cohort : cohorts_) {
            auto & source = *cohort.cpu;
            auto & arena = *cohort.arena;
            MixedMoePool pool = source;
            pool.local_experts = arena.slots;
            pool.expert_local = mfq_tensor_backend::full(
                {cpu_->n_experts}, -1,
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kInt32));
            size_t field = 0;
            if (pool.family == MixedMoeFamily::Nint) {
                pool.nint.aligned_q8 = false;
                pool.nint.workspaces.clear();
                pool.nint.q_packed = arena.fields.at(field++);
                pool.nint.row_q_bits = arena.fields.at(field++);
                pool.nint.row_q_bit_offsets = arena.fields.at(field++);
                pool.nint.sub_scale = arena.fields.at(field++);
                pool.nint.sub_min = arena.fields.at(field++);
                pool.nint.neuron_scale = arena.fields.at(field++);
                pool.nint.neuron_min = arena.fields.at(field++);
                pool.nint.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.nint.shape.empty()) {
                    pool.nint.shape[0] = pool.nint.out;
                }
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                pool.q8_zero.workspaces.clear();
                pool.q8_zero.q_packed = arena.fields.at(field++);
                pool.q8_zero.q8_zero_scale = arena.fields.at(field++);
                pool.q8_zero.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.q8_zero.shape.empty()) {
                    pool.q8_zero.shape[0] = pool.q8_zero.out;
                }
            } else if (pool.family == MixedMoeFamily::Mxfp4) {
                pool.mxfp4.values = arena.fields.at(field++);
                pool.mxfp4.scales = arena.fields.at(field++);
                pool.mxfp4.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
            } else if (pool.family == MixedMoeFamily::Nvq) {
                pool.nvq.workspaces.clear();
                pool.nvq.indices_packed = arena.fields.at(field++);
                pool.nvq.aux_packed = arena.fields.at(field++);
                pool.nvq.sub_scale_packed = arena.fields.at(field++);
                pool.nvq.neuron_scale = arena.fields.at(field++);
                pool.nvq.codebook =
                    copy_cpu_weight_to_cuda(source.nvq.codebook);
                pool.nvq.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.nvq.shape.empty()) {
                    pool.nvq.shape[0] = pool.nvq.out;
                }
            } else {
                pool.nepq.indices_packed = arena.fields.at(field++);
                pool.nepq.aux_packed = arena.fields.at(field++);
                pool.nepq.state_packed = arena.fields.at(field++);
                pool.nepq.neuron_scale = arena.fields.at(field++);
                pool.nepq.bank_ids = arena.fields.at(field++);
                pool.nepq.table_pool =
                    copy_cpu_weight_to_cuda(source.nepq.table_pool);
                pool.nepq.grouped_table_pool =
                    copy_cpu_weight_to_cuda(
                        source.nepq.grouped_table_pool);
                pool.nepq.rotation_signs =
                    copy_cpu_weight_to_cuda(
                        source.nepq.rotation_signs);
                if (pool.nepq.residual) {
                    pool.nepq.residual_codebook =
                        copy_cpu_weight_to_cuda(
                            source.nepq.residual_codebook);
                    pool.nepq.residual_first =
                        arena.fields.at(field++);
                    pool.nepq.residual_second =
                        arena.fields.at(field++);
                }
                pool.nepq.n_experts = arena.slots;
            }
            if (field != arena.fields.size()) {
                throw std::runtime_error(
                    "MoE cache active field count does not match its arena");
            }
            cohort.active = std::move(pool);
            active_->pools.push_back(cohort.active);
        }
    }

    void invalidate(int cohort_index, int expert, int slot) {
        if (cohort_index < 0 ||
            cohort_index >= static_cast<int>(cohorts_.size()) ||
            expert < 0 || expert >= cpu_->n_experts) {
            throw std::runtime_error(
                "invalid MoE cache eviction key");
        }
        auto & cohort =
            cohorts_.at(static_cast<size_t>(cohort_index));
        if (cohort.host_map[static_cast<size_t>(expert)] == slot) {
            cohort.host_map[static_cast<size_t>(expert)] = -1;
            cohort.map_dirty = true;
        }
    }

    std::vector<int32_t> route_experts(
            const MoeRoutePlan & route) const {
        if (!route.host_unique_experts) {
            route.host_unique_experts =
                std::make_shared<std::vector<int32_t>>(
                    cache_->read_route_experts(
                        route, cpu_->n_experts));
        }
        return *route.host_unique_experts;
    }

    void begin_prefetch(const MoeRoutePlan & route) {
        if (use_full_projection(route)) return;
        cache_->begin_route_experts(route, cpu_->n_experts);
    }

    bool use_full_projection(const MoeRoutePlan & route) const {
        return !range_store_ && route.ids.size(0) > 8;
    }

    mfq_tensor_backend::Tensor forward(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        if (use_full_projection(route)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime(execution.config);
            return staged.forward(execution, x, route);
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime(execution.config);
            return staged.forward(execution, x, route);
        }
        auto output = active_->forward(execution, x, route);
        cache_->record_compute_use();
        return output;
    }

    mfq_tensor_backend::Tensor forward_prequantized(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        if (use_full_projection(route)) {
            throw std::runtime_error(
                "cached prequantized activation reuse only supports decode-sized routes");
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime(execution.config);
            return staged.forward(execution, x, route);
        }
        auto output = active_->forward(execution, x, route, true);
        cache_->record_compute_use();
        return output;
    }

    void prefetch(const MoeRoutePlan & route) {
        if (use_full_projection(route)) return;
        (void)cache_->prepare(
            *this, route_experts(route), true);
    }

    mfq_tensor_backend::Tensor forward_glu_output(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) {
        if (use_full_projection(route)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime(execution.config);
            return staged.forward_glu_output(execution, x, route, gelu);
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime(execution.config);
            return staged.forward_glu_output(execution, x, route, gelu);
        }
        auto output = active_->forward_glu_output(
            execution, x, route, gelu);
        cache_->record_compute_use();
        return output;
    }

    mfq_tensor_backend::Tensor forward_glu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            bool gelu) {
        auto activation = gelu
            ? moe_geglu_split_cuda(gate_up)
            : moe_swiglu_split_cuda(gate_up);
        return forward(execution, activation, route);
    }

    mfq_tensor_backend::Tensor forward_clamped_swiglu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            double limit) {
        if (use_full_projection(route)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_cpu_mixed_moe(cpu_, execution.config);
            return staged.forward_clamped_swiglu(
                execution, gate_up, route, limit);
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_cpu_mixed_moe(cpu_, execution.config);
            return staged.forward_clamped_swiglu(
                execution, gate_up, route, limit);
        }
        auto output =
            active_->forward_clamped_swiglu(
                execution, gate_up, route, limit);
        cache_->record_compute_use();
        return output;
    }

    bool supports_clamped_swiglu() const {
        return cpu_->supports_clamped_swiglu();
    }

    static bool prefetch_bundle(
            const std::vector<std::shared_ptr<MoeCachedSource>> & sources,
            const MoeRoutePlan & route) {
        if (sources.empty()) return false;
        auto & first = *sources.front();
        if (first.use_full_projection(route)) return false;
        std::vector<MoeCachedSource *> raw_sources;
        raw_sources.reserve(sources.size());
        for (const auto & source : sources) {
            if (!source || source->cache_ != first.cache_ ||
                    source->n_experts() != first.n_experts() ||
                    source->layer_id() != first.layer_id() ||
                    source->use_full_projection(route)) {
                return false;
            }
            raw_sources.push_back(source.get());
        }
        if (raw_sources.size() == 3 &&
                raw_sources[0]->projection_role_ == "gate" &&
                raw_sources[1]->projection_role_ == "up" &&
                raw_sources[2]->projection_role_ == "down" &&
                raw_sources[2]->range_store_) {
            return first.cache_->prepare_bundle_deferred(
                {raw_sources[0], raw_sources[1]},
                *raw_sources[2],
                first.route_experts(route));
        }
        if (raw_sources.size() == 2 &&
                raw_sources[0]->projection_role_ == "gate_up" &&
                raw_sources[1]->projection_role_ == "down" &&
                raw_sources[1]->range_store_) {
            return first.cache_->prepare_bundle_deferred(
                {raw_sources[0]},
                *raw_sources[1],
                first.route_experts(route));
        }
        return first.cache_->prepare_bundle(
            raw_sources, first.route_experts(route));
    }

private:
    friend class MoeExpertCache;

    std::shared_ptr<MixedMoeRuntime> fallback_runtime() {
        if (!range_store_) return cpu_;
        std::lock_guard<std::mutex> guard(fallback_mutex_);
        if (!fallback_cpu_) {
            fallback_cpu_ = make_mixed_moe_runtime(
                unpack_mfe(range_store_->read_blob()), false);
        }
        return fallback_cpu_;
    }

    MfeWeight stage_fallback_runtime(
            const CudaExecutionConfig& config) {
        auto runtime = fallback_runtime();
        return stage_cpu_mixed_moe(runtime, config);
    }

    MoeExpertCache * cache_ = nullptr;
    int id_ = -1;
    std::string name_;
    int layer_id_ = -1;
    std::string projection_role_;
    std::shared_ptr<MixedMoeRuntime> cpu_;
    std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> range_store_;
    std::mutex fallback_mutex_;
    std::shared_ptr<MixedMoeRuntime> fallback_cpu_;
    std::vector<MoeCachedCohort> cohorts_;
    std::vector<int> expert_to_cohort_;
    std::vector<int> expert_to_local_;
    std::shared_ptr<MixedMoeRuntime> active_;
};

