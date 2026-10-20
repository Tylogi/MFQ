#pragma once

#include "quant/nint_rows.h"
#include "quant/nvq_rows.h"

#include <memory>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq {
class HostParallelPool;
class MfeQuantRangeUnsupported : public std::runtime_error {
public: using std::runtime_error::runtime_error;
};
struct QuantFieldTarget {std::uint8_t* data=nullptr;std::size_t bytes=0;};
// Convert a compact canonical NINT row range into row-order runtime fields.
// Targets: packed q (+8 guard bytes), q widths, int64 bit offsets, unpacked
// subgroup scale/min, and raw FP16 neuron scale/min.
void copy_nint_runtime_fields(const std::vector<std::uint8_t>& blob,
    const std::array<QuantFieldTarget,7>& targets);

struct MfeQuantExpert {
    std::string dtype;
    std::vector<std::uint8_t> payload;
};

// Canonical mixed NINT/NVQ/MXFP4 range source. Its index owns only selectors,
// codebooks and expert ownership; no expert payload is retained implicitly.
class MfeQuantExpertStore {
public:
    using Read=NintRows::Read;
    MfeQuantExpertStore(std::size_t bytes,Read read,NintRows::ReadBatch batch={});
    int num_experts() const noexcept { return experts_; }
    int out_per_expert() const noexcept { return output_; }
    int neuron_len() const noexcept { return width_; }
    std::size_t pool_count() const noexcept { return pools_.size(); }
    const std::vector<std::int32_t>& pool_expert_ids(std::size_t pool) const { return pools_.at(pool).ids; }
    int expert_pool(int expert) const;
    int expert_local(int expert) const;
    std::uint64_t expert_values_bits(int expert) const;
    std::size_t index_nbytes() const noexcept;
    const std::string& expert_dtype(int expert) const;
    std::size_t expert_payload_nbytes(int expert) const;
    MfeQuantExpert read_expert(int expert) const;
    std::optional<NvqRows::Fields> read_nvq_fields(int expert) const;
    bool copy_nvq_fields(int expert,const std::array<NvqRows::FieldTarget,4>& targets,
                         std::size_t* source_bytes=nullptr) const;
    // Adjacent pool rows can fill a contiguous runtime packet together.
    // Other selections return false without modifying the destinations.
    bool copy_nvq_fields(const std::vector<int>& experts,const std::array<NvqRows::FieldTarget,4>& targets,
                         std::size_t* source_bytes=nullptr) const;
    bool copy_nint_fields(int expert,const std::array<QuantFieldTarget,7>& targets,
                          std::size_t* source_bytes=nullptr) const;
    bool copy_mxfp4_fields(int expert,const std::array<QuantFieldTarget,2>& targets,
                          std::size_t* source_bytes=nullptr) const;
    // Startup-only sequential pool read; the temporary canonical pool is
    // released before advancing to the next pool. Visitors own each leaf.
    // An optional executor slices immutable pool bytes concurrently; callers
    // must provide a thread-safe visitor and join before returning.
    void visit_pool_experts(std::size_t pool,
        const std::function<void(int, MfeQuantExpert)>& visitor,
        HostParallelPool* workers=nullptr,int threads=1) const;
    std::size_t payload_nbytes() const noexcept { return bytes_; }
private:
    struct Pool {
        std::string dtype;
        std::vector<std::int32_t> ids;
        std::shared_ptr<mfq::NintRows> nint;
        std::shared_ptr<mfq::NvqRows> nvq;
        // MXT1 stores field-major rows; retain offsets, never pool payloads.
        std::size_t mx_values=0, mx_scales=0;
        std::size_t offset=0, bytes=0;
    };
    struct Expert { int pool=-1,local=-1; };
    Read read_;
    NintRows::ReadBatch read_batch_;
    std::size_t bytes_=0;
    int experts_=0,output_=0,width_=0;
    std::vector<Pool> pools_;
    std::vector<Expert> owners_;
};
} // namespace mfq
