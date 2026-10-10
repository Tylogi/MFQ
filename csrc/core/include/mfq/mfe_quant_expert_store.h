#pragma once

#include "mfq/nint_rows.h"
#include "mfq/nvq_rows.h"

#include <memory>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq {
class MfeQuantRangeUnsupported : public std::runtime_error {
public: using std::runtime_error::runtime_error;
};
struct MfeQuantExpert {
    std::string dtype;
    std::vector<std::uint8_t> payload;
};

// Canonical mixed NINT/NVQ range source. Its index owns only selectors,
// codebooks and expert ownership; no expert payload is retained implicitly.
class MfeQuantExpertStore {
public:
    using Read=NintRows::Read;
    MfeQuantExpertStore(std::size_t bytes,Read read);
    int num_experts() const noexcept { return experts_; }
    int out_per_expert() const noexcept { return output_; }
    int neuron_len() const noexcept { return width_; }
    std::size_t pool_count() const noexcept { return pools_.size(); }
    const std::vector<std::int32_t>& pool_expert_ids(std::size_t pool) const { return pools_.at(pool).ids; }
    int expert_pool(int expert) const;
    std::uint64_t expert_values_bits(int expert) const;
    std::size_t index_nbytes() const noexcept;
    const std::string& expert_dtype(int expert) const;
    std::size_t expert_payload_nbytes(int expert) const;
    MfeQuantExpert read_expert(int expert) const;
    // Startup-only sequential pool read; the temporary canonical pool is
    // released before advancing to the next pool. Visitors own each leaf.
    void visit_pool_experts(std::size_t pool,
        const std::function<void(int, MfeQuantExpert)>& visitor) const;
    std::size_t payload_nbytes() const noexcept { return bytes_; }
private:
    struct Pool {
        std::string dtype;
        std::vector<std::int32_t> ids;
        std::shared_ptr<mfq::NintRows> nint;
        std::shared_ptr<mfq::NvqRows> nvq;
        std::size_t offset=0, bytes=0;
    };
    struct Expert { int pool=-1,local=-1; };
    Read read_;
    std::size_t bytes_=0;
    int experts_=0,output_=0,width_=0;
    std::vector<Pool> pools_;
    std::vector<Expert> owners_;
};
} // namespace mfq
