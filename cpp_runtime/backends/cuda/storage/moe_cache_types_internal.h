#pragma once

#include "storage/moe_expert_cache.h"

#include "cuda_execution.h"
#include "moe.h"
#include "mfe_expert_store.h"
#include "moe_quant_range_source.h"
#include "moe_cache_policy.h"
#include "moe_cache_transfer.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <iostream>
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

static MixedMoePool moe_replace_quant_fields(MixedMoePool pool,const std::vector<mfq_tensor_backend::Tensor>& fields) {
    if(pool.family==MixedMoeFamily::Nint) {
        if(fields.size()!=7)throw std::invalid_argument("NINT RAM complement layout mismatch");
        pool.nint.q_packed=fields[0];pool.nint.row_q_bits=fields[1];pool.nint.row_q_bit_offsets=fields[2];
        pool.nint.sub_scale=fields[3];pool.nint.sub_min=fields[4];pool.nint.neuron_scale=fields[5];pool.nint.neuron_min=fields[6];
    }else if(pool.family==MixedMoeFamily::Nvq) {
        if(fields.size()!=4)throw std::invalid_argument("NVQ RAM complement layout mismatch");
        pool.nvq.indices_packed=fields[0];pool.nvq.aux_packed=fields[1];pool.nvq.sub_scale_packed=fields[2];pool.nvq.neuron_scale=fields[3];
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
    int64_t hybrid_calls = 0, hybrid_cpu_experts = 0, hybrid_gpu_experts = 0;
    int64_t ram_pcie_experts=0,ram_pcie_bytes=0;
    int64_t pipeline_serves=0,pipeline_route_wait_ns=0,pipeline_plan_ns=0;
    int64_t pipeline_two_stage_serves=0;
    int64_t pipeline_transfer_cache_hits=0,pipeline_transfer_cache_misses=0,pipeline_transfer_cache_saved_bytes=0;
    int64_t pipeline_fetch_ns=0,pipeline_cpu_ns=0;
    int64_t pipeline_window_serves=0,pipeline_window_route_ns=0,pipeline_window_plan_ns=0;
    int64_t pipeline_window_fetch_ns=0,pipeline_window_cpu_ns=0,pipeline_window_gpu_ns=0;
    int64_t pipeline_dma_copies=0,pipeline_window_dma_copies=0,pipeline_window_dma_bytes=0;
    int64_t pipeline_direct_ram_bytes=0,pipeline_direct_ram_copies=0,pipeline_staged_ram_bytes=0;
    int64_t pipeline_mapped_copy_serves=0,pipeline_mapped_copy_bytes=0,pipeline_mapped_copy_descriptors=0;
    int64_t pipeline_mapped_overlap_serves=0;
    int64_t pipeline_phased_transfer_serves=0,pipeline_gate_up_dma_bytes=0,pipeline_down_dma_bytes=0;
    int64_t pipeline_gate_up_primary_down_missing_experts=0,pipeline_gate_up_primary_down_missing_positions=0;
    int64_t pipeline_early_gate_up_positions=0,pipeline_early_gate_up_enabled=0;
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
