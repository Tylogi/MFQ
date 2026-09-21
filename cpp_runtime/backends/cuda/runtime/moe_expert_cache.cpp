#include "moe_expert_cache.h"

#include "moe.h"
#include "quant_linear.h"
#include "mfe_expert_store.h"
#include "moe_cache_policy.h"
#include "moe_cache_transfer.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using mfq_tensor_backend::indexing::Slice;

std::shared_ptr<MoeExpertCache> g_moe_expert_cache;
struct MoeCacheTransfer {
    const uint8_t * source = nullptr;
    uint8_t * destination = nullptr;
    int64_t nbytes = 0;
    bool packed_weight = false;
    const uint8_t * mapped_source = nullptr;
    const mfq::cuda::MfeMxfp4ExpertStore * range_store = nullptr;
    const mfq::cuda::MfeMxfp4ExpertPart * range_part = nullptr;
};

struct MoeCacheNewLease {
    mfq::MoeCacheSlotBook * book = nullptr;
    mfq::MoeCacheKey key;
    int slot = -1;
    uint64_t generation = 0;
};

struct MoeCachedCohort;
class MoeCachedSource;

struct MoeCacheFieldLayout {
    mfq_tensor_backend::ScalarType scalar_type =
        mfq_tensor_backend::kUInt8;
    std::vector<int64_t> slot_shape;
    int64_t elements = 0;
    int64_t element_size = 0;
};

static int64_t tensor_nbytes(const mfq_tensor_backend::Tensor & value) {
    return value.defined()
        ? value.numel() * static_cast<int64_t>(value.element_size())
        : 0;
}

static std::vector<mfq_tensor_backend::Tensor> moe_cache_fields(
        const MixedMoePool & pool) {
    if (pool.family == MixedMoeFamily::Nint) {
        return {
            pool.nint.q_packed,
            pool.nint.row_q_bits,
            pool.nint.row_q_bit_offsets,
            pool.nint.sub_scale,
            pool.nint.sub_min,
            pool.nint.neuron_scale,
            pool.nint.neuron_min,
        };
    }
    if (pool.family == MixedMoeFamily::Nint8Zero) {
        return {
            pool.q8_zero.q_packed,
            pool.q8_zero.q8_zero_scale,
        };
    }
    if (pool.family == MixedMoeFamily::Mxfp4) {
        return {pool.mxfp4.values, pool.mxfp4.scales};
    }
    if (pool.family == MixedMoeFamily::Tpq) {
        return {pool.tpq.packed};
    }
    if (pool.family == MixedMoeFamily::Nvq) {
        return {
            pool.nvq.indices_packed,
            pool.nvq.aux_packed,
            pool.nvq.sub_scale_packed,
            pool.nvq.neuron_scale,
        };
    }
    std::vector<mfq_tensor_backend::Tensor> fields{
        pool.nepq.indices_packed,
        pool.nepq.aux_packed,
        pool.nepq.state_packed,
        pool.nepq.neuron_scale,
        pool.nepq.bank_ids,
    };
    if (pool.nepq.residual) {
        fields.push_back(pool.nepq.residual_first);
        fields.push_back(pool.nepq.residual_second);
    }
    return fields;
}

static std::vector<MoeCacheFieldLayout> moe_cache_field_layouts(
        const MixedMoePool & pool) {
    const auto fields = moe_cache_fields(pool);
    std::vector<MoeCacheFieldLayout> result;
    result.reserve(fields.size());
    for (const auto & field : fields) {
        if (!field.defined() || !field.is_cpu() || !field.is_contiguous() ||
                field.dim() < 1 || pool.local_experts <= 0 ||
                field.size(0) % pool.local_experts != 0 ||
                field.numel() % pool.local_experts != 0) {
            throw std::runtime_error(
                "MoE cache source fields must be contiguous, expert-major CPU tensors");
        }
        auto shape = field.sizes().vec();
        shape[0] /= pool.local_experts;
        result.push_back({
            field.scalar_type(),
            std::move(shape),
            field.numel() / pool.local_experts,
            static_cast<int64_t>(field.element_size()),
        });
    }
    return result;
}

static void validate_nepq_expert_boundaries(
        const MixedMoePool & pool,
        int out_per_expert,
        int neuron_len) {
    if (pool.family != MixedMoeFamily::Nepq) return;
    const int index_bits =
        pool.nepq.format == 9 ? 6 :
        pool.nepq.format == 7 ? 7 :
        pool.nepq.format == 8 ? 9 :
        pool.nepq.format == 1 ? 11 : 0;
    const int aux_bits =
        pool.nepq.format == 8 || pool.nepq.format == 1 ? 1 : 0;
    const int nvec = neuron_len / 8;
    const int ng = (neuron_len + 23) / 24;
    const int nsuper = (ng + 3) / 4;
    const int rows = pool.local_experts * out_per_expert;
    const std::vector<int64_t> bits{
        static_cast<int64_t>(out_per_expert) * nvec * index_bits,
        static_cast<int64_t>(out_per_expert) * ng * aux_bits,
        static_cast<int64_t>(out_per_expert) * ng *
            pool.nepq.state_bits,
        static_cast<int64_t>(out_per_expert) * 32,
        static_cast<int64_t>(out_per_expert) * nsuper * 8,
    };
    if (index_bits == 0 ||
        std::any_of(bits.begin(), bits.end(), [](int64_t value) {
            return value % 8 != 0;
        })) {
        throw std::runtime_error(
            "NEPQ expert payload is not byte aligned for GPU caching");
    }
    if (pool.nepq.residual) {
        const int blocks =
            (nvec + pool.nepq.residual_block_vectors - 1) /
            pool.nepq.residual_block_vectors;
        if (!pool.nepq.residual_codebook.defined() ||
                !pool.nepq.residual_codebook.is_cpu() ||
                !pool.nepq.residual_codebook.is_contiguous() ||
                pool.nepq.residual_codebook.scalar_type() != mfq_tensor_backend::kFloat16 ||
                pool.nepq.residual_codebook.dim() != 2 ||
                pool.nepq.residual_codebook.size(0) != 1024 ||
                pool.nepq.residual_codebook.size(1) != 8 ||
                !pool.nepq.residual_first.defined() ||
                !pool.nepq.residual_first.is_cpu() ||
                !pool.nepq.residual_first.is_contiguous() ||
                pool.nepq.residual_first.scalar_type() != mfq_tensor_backend::kInt16 ||
                pool.nepq.residual_first.dim() != 2 ||
                pool.nepq.residual_first.size(0) != rows ||
                pool.nepq.residual_first.size(1) != blocks ||
                !pool.nepq.residual_second.defined() ||
                !pool.nepq.residual_second.is_cpu() ||
                !pool.nepq.residual_second.is_contiguous() ||
                pool.nepq.residual_second.scalar_type() != mfq_tensor_backend::kInt16 ||
                pool.nepq.residual_second.dim() != 2 ||
                pool.nepq.residual_second.size(0) != rows ||
                pool.nepq.residual_second.size(1) != blocks) {
            throw std::runtime_error(
                "NEPQ-A residual fields are incompatible with GPU caching");
        }
    }
}

static void validate_tpq_expert_boundaries(
        const MixedMoePool & pool,
        int out_per_expert,
        int neuron_len) {
    if (pool.family != MixedMoeFamily::Tpq) return;
    if (pool.tpq.vector_size <= 0) {
        throw std::runtime_error("TPQ-PQ cache vector size must be positive");
    }
    const int64_t vectors = neuron_len / pool.tpq.vector_size;
    const int64_t bits_per_expert =
        static_cast<int64_t>(out_per_expert) * vectors *
        pool.tpq.index_bits;
    if (pool.tpq.int4 ||
            neuron_len % pool.tpq.vector_size != 0 ||
            pool.tpq.index_bits < 8 || pool.tpq.index_bits > 16 ||
            bits_per_expert % 8 != 0 ||
            !pool.tpq.packed.defined() || !pool.tpq.packed.is_cpu() ||
            !pool.tpq.packed.is_contiguous() ||
            pool.tpq.packed.scalar_type() != mfq_tensor_backend::kUInt8 ||
            pool.tpq.packed.numel() !=
                pool.local_experts * bits_per_expert / 8 ||
            !pool.tpq.codebook.defined() || !pool.tpq.codebook.is_cpu() ||
            !pool.tpq.codebook.is_contiguous() ||
            pool.tpq.codebook.scalar_type() != mfq_tensor_backend::kFloat32 ||
            pool.tpq.codebook.dim() != 2 ||
            pool.tpq.codebook.size(0) <= 1 ||
            pool.tpq.codebook.size(1) != pool.tpq.vector_size) {
        throw std::runtime_error(
            "TPQ-PQ expert fields are incompatible with GPU caching");
    }
}

static std::string moe_cache_signature(
        const MixedMoePool & pool,
        int out_per_expert,
        int neuron_len,
        const std::vector<MoeCacheFieldLayout> & layouts) {
    std::ostringstream stream;
    stream << static_cast<int>(pool.family)
           << ":o" << out_per_expert
           << ":k" << neuron_len;
    if (pool.family == MixedMoeFamily::Nint) {
        stream << ":b" << pool.nint.bits
               << ":g" << pool.nint.gs
               << ":n" << pool.nint.ng
               << ":s" << pool.nint.q_expert_stride
               << ":v2";
    } else if (pool.family == MixedMoeFamily::Nint8Zero) {
        stream << ":g32:n" << pool.q8_zero.ng;
    } else if (pool.family == MixedMoeFamily::Mxfp4) {
        stream << ":mx4";
    } else if (pool.family == MixedMoeFamily::Tpq) {
        stream << ":tpq:v" << pool.tpq.vector_size
               << ":b" << pool.tpq.index_bits;
    } else if (pool.family == MixedMoeFamily::Nvq) {
        stream << ":f" << pool.nvq.format
               << ":kf" << pool.nvq.kernel_format
               << ":s" << pool.nvq.sub_bits
               << ":g" << pool.nvq.gs
               << ":n" << pool.nvq.ng
               << ":sm" << pool.nvq.sign_mode;
    } else {
        stream << ":f" << pool.nepq.format
               << ":s" << pool.nepq.state_bits
               << ":g" << pool.nepq.ng
               << ":r" << pool.nepq.rotation_block
               << ":a" << (pool.nepq.residual ? 1 : 0);
        if (pool.nepq.residual) {
            stream << ":p" << pool.nepq.residual_position_bits
                   << ":v" << pool.nepq.residual_block_vectors;
        }
    }
    for (const auto & layout : layouts) {
        stream << ":" << static_cast<int>(layout.scalar_type)
               << "x" << layout.elements;
    }
    return stream.str();
}

struct MoeGpuArena {
    std::string signature;
    int64_t slot_bytes = 0;
    int minimum_slots = 0;
    int registered_experts = 0;
    int slots = 0;
    std::vector<MoeCacheFieldLayout> layouts;
    std::vector<mfq_tensor_backend::Tensor> fields;
    std::unique_ptr<mfq::MoeCacheSlotBook> book;
};

struct MoePinnedStage {
    mfq_tensor_backend::Tensor host;
    mfq_tensor_backend::Tensor device;
    cudaEvent_t done = nullptr;
    bool pending = false;
};

struct MoePendingRangeRead {
    mfq_tensor_backend::Tensor host;
    std::vector<MoeCacheTransfer> transfers;
    std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> held_slots;
    std::vector<MoeCacheNewLease> new_leases;
    mfq::cuda::MfeMxfp4ReadTicket ticket;
    bool replaced_occupied = false;
};

struct MoeCacheStats {
    int64_t demand_hits = 0;
    int64_t demand_misses = 0;
    int64_t prefetch_hits = 0;
    int64_t prefetch_misses = 0;
    int64_t evictions = 0;
    int64_t h2d_bytes = 0;
    int64_t route_d2h_bytes = 0;
    int64_t full_projection_fallbacks = 0;
    int64_t h2d_submissions = 0;
    int64_t h2d_descriptors = 0;
    int64_t mapped_gather_bytes = 0;
    int64_t mapped_gather_submissions = 0;
    int64_t mapped_gather_descriptors = 0;
    int64_t range_read_bytes = 0;
    int64_t range_read_calls = 0;
    int64_t range_file_opens = 0;
    int64_t range_read_nanoseconds = 0;
    int64_t range_overlap_batches = 0;
    int64_t range_overlap_wait_nanoseconds = 0;
};

class MoeExpertCache : public std::enable_shared_from_this<MoeExpertCache> {
public:
    explicit MoeExpertCache(int64_t budget_bytes)
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
        const char * mapped = std::getenv("MFQ_MOE_MAPPED_GATHER");
        mapped_gather_enabled_ =
            mapped != nullptr && std::atoi(mapped) != 0;
        const char * blocks =
            std::getenv("MFQ_MOE_MAPPED_COPY_BLOCKS");
        if (blocks != nullptr) {
            mapped_copy_blocks_ = std::max(
                4, std::min(128, std::atoi(blocks)));
        }
        int range_workers = 8;
        const char * workers =
            std::getenv("MFQ_MOE_SSD_IO_WORKERS");
        if (workers != nullptr) {
            range_workers = std::max(
                1, std::min(64, std::atoi(workers)));
        }
        range_read_pool_ =
            std::make_unique<mfq::cuda::MfeMxfp4ReadPool>(range_workers);
        const char * disable_overlap =
            std::getenv("MFQ_DISABLE_MOE_SSD_OVERLAP");
        range_overlap_enabled_ =
            disable_overlap == nullptr || std::atoi(disable_overlap) == 0;
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
        validate_tpq_expert_boundaries(
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

    void print_stats(std::ostream & stream) const {
        stream << "moe_cache_stats"
               << " budget_bytes=" << budget_bytes_
               << " allocated_bytes=" << allocated_bytes_
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
};

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
            } else if (pool.family == MixedMoeFamily::Tpq) {
                pool.tpq.packed = arena.fields.at(field++);
                pool.tpq.codebook =
                    copy_cpu_weight_to_cuda(source.tpq.codebook);
                pool.tpq.out =
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
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        if (use_full_projection(route)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime();
            return staged.forward(x, route);
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime();
            return staged.forward(x, route);
        }
        auto output = active_->forward(x, route);
        cache_->record_compute_use();
        return output;
    }

    mfq_tensor_backend::Tensor forward_prequantized(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        if (use_full_projection(route)) {
            throw std::runtime_error(
                "cached prequantized activation reuse only supports decode-sized routes");
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime();
            return staged.forward(x, route);
        }
        auto output = active_->forward(x, route, true);
        cache_->record_compute_use();
        return output;
    }

    void prefetch(const MoeRoutePlan & route) {
        if (use_full_projection(route)) return;
        (void)cache_->prepare(
            *this, route_experts(route), true);
    }

    mfq_tensor_backend::Tensor forward_glu_output(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) {
        if (use_full_projection(route)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime();
            return staged.forward_glu_output(x, route, gelu);
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime();
            return staged.forward_glu_output(x, route, gelu);
        }
        auto output = active_->forward_glu_output(x, route, gelu);
        cache_->record_compute_use();
        return output;
    }

    mfq_tensor_backend::Tensor forward_glu(
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            bool gelu) {
        auto activation = gelu
            ? moe_geglu_split_cuda(gate_up)
            : moe_swiglu_split_cuda(gate_up);
        return forward(activation, route);
    }

    mfq_tensor_backend::Tensor forward_clamped_swiglu(
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            double limit) {
        if (use_full_projection(route)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_cpu_mixed_moe(cpu_);
            return staged.forward_clamped_swiglu(
                gate_up, route, limit);
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_cpu_mixed_moe(cpu_);
            return staged.forward_clamped_swiglu(
                gate_up, route, limit);
        }
        auto output =
            active_->forward_clamped_swiglu(
                gate_up, route, limit);
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

    MfeWeight stage_fallback_runtime() {
        auto runtime = fallback_runtime();
        return stage_cpu_mixed_moe(runtime);
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
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        return source->forward(x, route);
    };
    result.mixed_prequantized_forward = [source](
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        return source->forward_prequantized(x, route);
    };
    result.mixed_glu_output_forward = [source](
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) {
        return source->forward_glu_output(
            x, route, gelu);
    };
    result.mixed_glu_forward = [source](
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            bool gelu) {
        return source->forward_glu(
            gate_up, route, gelu);
    };
    if (source->supports_clamped_swiglu()) {
        result.mixed_clamped_swiglu_forward = [source](
                mfq_tensor_backend::Tensor gate_up,
                const MoeRoutePlan & route,
                double limit) {
            return source->forward_clamped_swiglu(
                gate_up, route, limit);
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

MfeWeight load_mfe_gpu(
        const mfq::ModelSource & mfq, const std::string & name,
        bool cacheable,
        int layer_id,
        const std::string & projection_role) {
    const char * disable_ranges =
        std::getenv("MFQ_DISABLE_MOE_SSD_RANGES");
    if (g_moe_expert_cache && cacheable &&
            !moe_parallel_config().enabled() &&
            (disable_ranges == nullptr || std::atoi(disable_ranges) == 0)) {
        const auto & record = require_tensor(mfq, name);
        try {
            auto store =
                std::make_shared<mfq::cuda::MfeMxfp4ExpertStore>(
                    mfq::cuda::MfqRecordRange{
                        name,
                        record.dtype,
                        {},
                        0,
                        record.nbytes,
                        [&mfq, name](
                                std::uint64_t offset,
                                std::span<std::uint8_t> destination) {
                            mfq.read_range_into(
                                name,
                                offset,
                                reinterpret_cast<std::byte*>(destination.data()),
                                destination.size());
                        },
                    });
            auto runtime = make_mxfp4_range_runtime(*store);
            auto source = g_moe_expert_cache->register_range_source(
                name,
                runtime,
                std::move(store),
                std::min(
                    g_moe_cache_registration_min_slots,
                    runtime->n_experts),
                layer_id,
                projection_role);
            return wrap_cached_moe_source(source, runtime);
        } catch (const mfq::cuda::MfeMxfp4Unsupported &) {
        }
    }
    auto cpu = load_mfe_cpu(mfq, name);
    const bool has_matrix_local_sq = std::any_of(
        cpu.pools.begin(), cpu.pools.end(), [](const MfeCpuPool & pool) {
            return pool.dtype == "MXFP4-SQ" ||
                mfq::fp8sq::is_dtype(pool.dtype);
        });
    if (moe_parallel_config().enabled()) {
        auto slices = plan_moe_expert_parallel_slices(
            cpu.n_experts, name);
        MfeWeight result;
        result.n_experts = cpu.n_experts;
        result.out_per_expert =
            cpu.out_per_expert;
        result.neuron_len =
            cpu.neuron_len;
        for (const auto & slice : slices) {
            auto shard =
                std::make_shared<MfeWeight>(
                    to_cuda_device_moe_expert_slice(
                        cpu, slice.begin,
                        slice.end,
                        slice.device));
            result.expert_parallel_shards.push_back({
                slice.device,
                slice.begin,
                slice.end,
                std::move(shard),
            });
        }
        return result;
    }
    if (g_moe_expert_cache && cacheable && !has_matrix_local_sq) {
        auto runtime =
            make_mixed_moe_runtime(cpu, false);
        auto source = g_moe_expert_cache->register_source(
            name, runtime,
            std::min(
                g_moe_cache_registration_min_slots,
                runtime->n_experts),
            layer_id,
            projection_role);
        return wrap_cached_moe_source(source, runtime);
    }
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(), [](const MfeCpuPool & pool) {
            return pool.dtype == "NINT";
        });
    return all_nint ? to_gpu_mfe(cpu) : to_gpu_mixed_moe(cpu);
}

std::shared_ptr<MixedMoeRuntime> load_mfe_cpu_offloaded(
        const mfq::ModelSource & mfq, const std::string & name) {
    return make_mixed_moe_runtime(load_mfe_cpu(mfq, name), false);
}
std::shared_ptr<MoeExpertCache> make_moe_expert_cache(std::int64_t bytes) {
    return std::make_shared<MoeExpertCache>(bytes);
}

bool moe_expert_cache_has_sources() {
    return g_moe_expert_cache && g_moe_expert_cache->has_sources();
}

bool moe_expert_cache_finalized() {
    return g_moe_expert_cache && g_moe_expert_cache->finalized();
}

void finalize_moe_expert_cache() {
    if (g_moe_expert_cache && !g_moe_expert_cache->finalized()) {
        g_moe_expert_cache->finalize();
    }
}

void print_moe_expert_cache_stats(std::ostream& output) {
    if (g_moe_expert_cache) {
        g_moe_expert_cache->print_stats(output);
    }
}

void set_moe_expert_cache_profile(mfq::MoeCacheProfile profile) {
    if (!g_moe_expert_cache) {
        throw std::runtime_error("MoE expert cache is not configured");
    }
    g_moe_expert_cache->set_profile(std::move(profile));
}
