#pragma once

#include "storage/moe_expert_cache.h"

#include "cuda_execution.h"
#include "moe.h"
#include "mfe_expert_store.h"
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

static auto moe_cache_field_refs(auto& pool) {
    using Field = decltype(&pool.nint.q_packed);
    if (pool.family == MixedMoeFamily::Nint) {
        return std::vector<Field>{
            &pool.nint.q_packed,
            &pool.nint.row_q_bits,
            &pool.nint.row_q_bit_offsets,
            &pool.nint.sub_scale,
            &pool.nint.sub_min,
            &pool.nint.neuron_scale,
            &pool.nint.neuron_min,
        };
    }
    if (pool.family == MixedMoeFamily::Nint8Zero) {
        return std::vector<Field>{
            &pool.q8_zero.q_packed,
            &pool.q8_zero.q8_zero_scale,
        };
    }
    if (pool.family == MixedMoeFamily::Mxfp4) {
        return std::vector<Field>{&pool.mxfp4.values, &pool.mxfp4.scales};
    }
    if (pool.family == MixedMoeFamily::Nvq) {
        return std::vector<Field>{
            &pool.nvq.indices_packed,
            &pool.nvq.aux_packed,
            &pool.nvq.sub_scale_packed,
            &pool.nvq.neuron_scale,
        };
    }
    std::vector<Field> fields{
        &pool.nepq.indices_packed,
        &pool.nepq.aux_packed,
        &pool.nepq.state_packed,
        &pool.nepq.neuron_scale,
        &pool.nepq.bank_ids,
    };
    if (pool.nepq.residual) {
        fields.push_back(&pool.nepq.residual_first);
        fields.push_back(&pool.nepq.residual_second);
    }
    return fields;
}

static std::vector<mfq_tensor_backend::Tensor> moe_cache_fields(
        const MixedMoePool& pool) {
    std::vector<mfq_tensor_backend::Tensor> fields;
    for (const auto* field : moe_cache_field_refs(pool)) fields.push_back(*field);
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
