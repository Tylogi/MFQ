#pragma once

#include "storage/moe_expert_cache.h"

#include "cuda_execution.h"
#include "moe.h"
#include "mfe_expert_store.h"
#include "moe_quant_range_source.h"
#include "moe_cache_policy.h"
#include "moe_cache_transfer.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using mfq_tensor_backend::indexing::Slice;

struct MoeCacheTransfer {
    const uint8_t * source = nullptr;
    uint8_t * destination = nullptr;
    int64_t nbytes = 0;
    bool packed_weight = false;
    const uint8_t * mapped_source = nullptr;
    const mfq::cuda::MfeMxfp4ExpertStore * range_store = nullptr;
    const mfq::cuda::MfeMxfp4ExpertPart * range_part = nullptr;
    std::shared_ptr<const void> source_owner;
};

struct MoeCacheNewLease {
    mfq::MoeCacheSlotBook * book = nullptr;
    mfq::MoeCacheKey key;
    int slot = -1;
    uint64_t generation = 0;
    // A direct loan view did not populate every field of this staging slot.
    // Keep the slot protected until the next route boundary, then discard it.
    bool aliased = false;
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

// Borrow immutable descriptors while the caller holds their pool/lease. A
// residency transaction revisits these fields several times; copying Tensors
// into a new vector each time adds allocation and shared-owner traffic.
struct MoeCacheFieldRefs {
    using Tensor=mfq_tensor_backend::Tensor;
    std::array<const Tensor*,7> fields{};
    std::size_t count=0;
    MoeCacheFieldRefs(std::initializer_list<const Tensor*> values):count(values.size()) {
        std::copy(values.begin(),values.end(),fields.begin());
    }
    struct Iterator {
        const Tensor* const* at;
        const Tensor& operator*() const{return **at;}
        Iterator& operator++(){++at;return *this;}
        bool operator!=(const Iterator& other)const{return at!=other.at;}
    };
    Iterator begin()const{return {fields.data()};}
    Iterator end()const{return {fields.data()+count};}
    const Tensor& operator[](std::size_t index)const{return *fields[index];}
    std::size_t size()const{return count;}
};
static MoeCacheFieldRefs moe_cache_field_refs(const MixedMoePool& pool) {
    if(pool.family==MixedMoeFamily::Nint)return {&pool.nint.q_packed,&pool.nint.row_q_bits,
        &pool.nint.row_q_bit_offsets,&pool.nint.sub_scale,&pool.nint.sub_min,&pool.nint.neuron_scale,&pool.nint.neuron_min};
    if(pool.family==MixedMoeFamily::Nint8Zero)return {&pool.q8_zero.q_packed,&pool.q8_zero.q8_zero_scale};
    if(pool.family==MixedMoeFamily::Mxfp4)return {&pool.mxfp4.values,&pool.mxfp4.scales};
    if(pool.family==MixedMoeFamily::Nvq)return {&pool.nvq.indices_packed,&pool.nvq.aux_packed,
        &pool.nvq.sub_scale_packed,&pool.nvq.neuron_scale};
    if(pool.nepq.residual)return {&pool.nepq.indices_packed,&pool.nepq.aux_packed,&pool.nepq.state_packed,
        &pool.nepq.neuron_scale,&pool.nepq.bank_ids,&pool.nepq.residual_first,&pool.nepq.residual_second};
    return {&pool.nepq.indices_packed,&pool.nepq.aux_packed,&pool.nepq.state_packed,&pool.nepq.neuron_scale,&pool.nepq.bank_ids};
}

static mfq_tensor_backend::Tensor moe_owned_host_view(
        const std::shared_ptr<mfq_tensor_backend::Tensor>& arena,std::size_t offset,
        const mfq_tensor_backend::Tensor& field) {
    namespace tb=mfq_tensor_backend;
    const auto count=static_cast<std::size_t>(tensor_nbytes(field));
    if(!arena || !arena->is_cpu() || offset>static_cast<std::size_t>(arena->numel()) ||
        count>static_cast<std::size_t>(arena->numel())-offset)
        throw std::out_of_range("RAM complement field exceeds its owned arena");
    auto* ptr=arena->data_ptr<uint8_t>()+offset;
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    auto storage=std::make_shared<tb::TensorStorage>();
    storage->owner=arena;storage->base=ptr;storage->bytes=count;storage->device=field.device();
    return tb::Tensor(storage,tb::make_contiguous_view(ptr,field.sizes(),field.scalar_type(),field.device()));
#else
    return tb::from_blob(ptr,field.sizes(),[arena](void*){},field.options());
#endif
}

template<class Fields>
static MixedMoePool moe_replace_quant_fields(MixedMoePool pool,const Fields& fields) {
    if(pool.family==MixedMoeFamily::Nint) {
        if(fields.size()!=7)throw std::invalid_argument("NINT RAM complement layout mismatch");
        pool.nint.q_packed=fields[0];pool.nint.row_q_bits=fields[1];pool.nint.row_q_bit_offsets=fields[2];
        pool.nint.sub_scale=fields[3];pool.nint.sub_min=fields[4];pool.nint.neuron_scale=fields[5];pool.nint.neuron_min=fields[6];
    }else if(pool.family==MixedMoeFamily::Nvq) {
        if(fields.size()!=4)throw std::invalid_argument("NVQ RAM complement layout mismatch");
        pool.nvq.indices_packed=fields[0];pool.nvq.aux_packed=fields[1];pool.nvq.sub_scale_packed=fields[2];pool.nvq.neuron_scale=fields[3];
    }else if(pool.family==MixedMoeFamily::Mxfp4) {
        if(fields.size()!=2)throw std::invalid_argument("MXFP4 RAM complement layout mismatch");
        pool.mxfp4.values=fields[0];pool.mxfp4.scales=fields[1];
    }else throw std::invalid_argument("unsupported canonical RAM complement format");
    return pool;
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
    std::optional<mfq::MoeCacheKey> prefill_backup_key;
    std::uint64_t prefill_backup_generation=0;
    int64_t slot_bytes = 0;
    int minimum_slots = 0;
    int registered_experts = 0;
    int slots = 0;
    std::vector<MoeCacheFieldLayout> layouts;
    std::vector<mfq_tensor_backend::Tensor> fields;
    // Decode descriptors can address heterogeneous experts in one byte arena.
    // The slot book belongs to the arena, so formats compete for the same slots.
    mfq_tensor_backend::Tensor packed;
    std::unique_ptr<mfq::MoeCacheSlotBook> book;
    uint8_t* field_pointer(int slot, std::size_t field,
                          const std::vector<int64_t>& bytes) const {
        if(!packed.defined())return static_cast<uint8_t*>(fields[field].data_ptr())+int64_t(slot)*bytes[field];
        int64_t offset=0;
        for(std::size_t f=0;f<field;++f)offset+=(bytes[f]+255)&~int64_t(255);
        return packed.data_ptr<uint8_t>()+int64_t(slot)*slot_bytes+offset;
    }
};

inline int64_t moe_packed_slot_bytes(const std::vector<int64_t>& fields) {
    int64_t result=0;
    for(auto bytes:fields)result+=(bytes+255)&~int64_t(255);
    return result;
}

inline bool moe_packed_transfer_stages_enabled() {
    const auto* value=std::getenv("MFQ_MOE_PACKED_TRANSFER_STAGE");
    return !value || value[0]!='0';
}

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
    int64_t hybrid_calls = 0, hybrid_cpu_experts = 0, hybrid_gpu_experts = 0;
    int64_t ram_pcie_experts=0,ram_pcie_bytes=0;
    int64_t pipeline_serves=0,pipeline_route_wait_ns=0,pipeline_plan_ns=0;
    int64_t pipeline_two_stage_serves=0;
    std::uint64_t pipeline_graph_captures=0;
    std::array<std::uint64_t,2> pipeline_multi_route_positions{},pipeline_multi_unique_routes{};
    int64_t pipeline_transfer_cache_hits=0,pipeline_transfer_cache_misses=0,pipeline_transfer_cache_saved_bytes=0;
    int64_t pipeline_fetch_ns=0,pipeline_cpu_ns=0;
    int64_t pipeline_host_copy_ns=0;
    int64_t pipeline_ssd_read_ns=0,pipeline_window_ssd_read_ns=0;
    int64_t pipeline_window_serves=0,pipeline_window_route_ns=0,pipeline_window_plan_ns=0;
    int64_t pipeline_window_fetch_ns=0,pipeline_window_cpu_ns=0,pipeline_window_gpu_ns=0;
    // Host dispatch plans for three/four-row windows; routed experts only.
    int64_t pipeline_small_window_groups[5]{},pipeline_small_window_hot_groups[5]{};
    int64_t pipeline_dma_copies=0,pipeline_window_dma_copies=0,pipeline_window_dma_bytes=0;
    int64_t pipeline_window_pcie_payload_bytes=0;
    int64_t pipeline_fused_transfer_copy_serves=0;
    int64_t pipeline_empty_transfer_serves=0;
    int64_t pipeline_dma_graph_launches=0;
    int64_t pipeline_dma_graph_fallbacks=0;
    int64_t pipeline_direct_ram_bytes=0,pipeline_direct_ram_copies=0,pipeline_staged_ram_bytes=0;
    int64_t pipeline_prefill_group_serves=0,pipeline_prefill_groups=0;
    int64_t pipeline_mapped_copy_serves=0,pipeline_mapped_copy_bytes=0,pipeline_mapped_copy_descriptors=0;
    int64_t pipeline_mapped_overlap_serves=0;
    int64_t pipeline_phased_transfer_serves=0,pipeline_gate_up_dma_bytes=0,pipeline_down_dma_bytes=0;
    int64_t pipeline_prefill_cpu_copy_jobs=0,pipeline_prefill_cpu_copy_join_ns=0;
    int64_t pipeline_gate_up_primary_down_missing_experts=0,pipeline_gate_up_primary_down_missing_positions=0;
    int64_t pipeline_early_gate_up_positions=0,pipeline_early_gate_up_enabled=0;
    int64_t pipeline_early_no_cpu_ready_serves=0;
    int64_t pipeline_gpu_only_cpu_wait_skips=0;
    int64_t pipeline_fixed_gpu_timing_skips=0;
    int64_t pipeline_transfer_stream_writes=0,pipeline_transfer_host_callbacks=0;
    int64_t pipeline_transfer_event_polls=0;
    int64_t pipeline_kv_direct_read_bytes=0,pipeline_kv_direct_read_fields=0;
    int64_t pipeline_kv_early_gu_serves=0;
    int64_t pipeline_gu_build_overlap_serves=0,pipeline_gu_build_overlap_copies=0;
    int64_t pipeline_ssd_gu_overlap_serves=0,pipeline_ssd_gu_overlap_bytes=0;
    int64_t pipeline_ssd_gu_staged_bytes=0;
    int64_t pipeline_ssd_gu_upload_batches=0;
    int64_t pipeline_ssd_down_overlap_bytes=0,pipeline_ssd_down_upload_batches=0,pipeline_ssd_down_staged_bytes=0;
    int64_t pipeline_ssd_down_early_max_batch=0,pipeline_ssd_down_early_max_bytes=0;
    int64_t pipeline_ssd_gu_compute_serves=0,pipeline_ssd_gate_up_wait_ns=0,pipeline_ssd_down_wait_ns=0;
    int64_t pipeline_kv_projection_hits=0,pipeline_kv_projection_misses=0,pipeline_kv_projection_fallbacks=0;
    int64_t pipeline_kv_projection_small_hit_bytes=0;
    int64_t pipeline_kv_projection_early_hits=0;
    int64_t pipeline_kv_gpu_resident_experts=0,pipeline_kv_gpu_resident_bytes=0;
    int64_t pipeline_kv_gpu_early_gu_experts=0;
    int64_t pipeline_reader_fence_reuses=0;
    int64_t pipeline_host_packet_serves=0,pipeline_host_packet_descriptor_bytes=0,pipeline_cached_phase_publishes=0;
    int64_t pipeline_direct_packet_serves=0,pipeline_direct_packet_batch_serves=0;
    int64_t pipeline_native_dma_batches=0,pipeline_native_dma_copies=0,pipeline_native_dma_fallbacks=0;
    int64_t pipeline_projection_dma_merges=0;
    int64_t pipeline_window_input_view_serves=0;
    int64_t pipeline_prefetch_experts=0,pipeline_prefetch_bytes=0,pipeline_prefetch_hit_bytes=0,pipeline_prefetch_busy_skips=0;
    int64_t pipeline_gpu_ns=0;
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
    int64_t gpu_demote_bytes = 0;
};

struct MoePreloadH2DTiming {
    cudaEvent_t begin=nullptr,end=nullptr;
    std::int64_t bytes=0;
    explicit MoePreloadH2DTiming(std::int64_t size):bytes(size) {
        MFQ_CUDA_CHECK(cudaEventCreate(&begin));
        const auto result=cudaEventCreate(&end);
        if (result!=cudaSuccess) { cudaEventDestroy(begin); begin=nullptr; MFQ_CUDA_CHECK(result); }
    }
    ~MoePreloadH2DTiming() { if (begin) cudaEventDestroy(begin); if (end) cudaEventDestroy(end); }
};
