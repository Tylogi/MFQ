#pragma once

#include "mfq_tensor_backend.h"
#include "kv_offload.h"
#include "mfq_cuda_kv_hot_cache.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mfq::cuda {
class KvExpertLoan;
// Placement applies to raw K/V. Indexer and kernel scratch remain GPU-local.
struct KvMemoryOptions {
    std::string placement = "auto";
    bool dynamic = false;
    bool budget_policy = false, ram_preallocated = true;
    std::size_t gpu_budget = 0;
    std::size_t gpu_limit = 0, ram_limit = 0;
    bool ram_limit_set = false;
    bool additional_request = false;
    int64_t auto_stream_threshold = 65536, resident_cells = 32768;
    std::string directory;
};
KvMemoryOptions kv_memory_options();

struct KvMemoryAllocation {
    struct Virtual;
    mfq_tensor_backend::Tensor keys, values, index, pooled;
    std::shared_ptr<KvOffloadStore> offload;
    std::shared_ptr<Virtual> virtual_keys, virtual_values;
    int64_t maximum = 0, heads = 0, key_width = 0, value_width = 0, index_width = 0, pool = 0, selected_columns = 0;
    mfq_tensor_backend::TensorOptions options;
    bool host = false, ready = false;
    HotKvCache hot;
    std::weak_ptr<KvExpertLoan> expert_loan;
    std::size_t raw_bytes() const;
    std::size_t index_bytes() const;
    std::size_t gpu_bytes() const;
    std::size_t ram_bytes() const;
    std::size_t growth_bytes(int64_t tokens) const;
    void ensure(int64_t tokens);
};

class KvMemoryManager {
public:
    explicit KvMemoryManager(KvMemoryOptions options = kv_memory_options());
    std::shared_ptr<KvMemoryAllocation> add(int64_t maximum, int64_t heads,
        int64_t key_width, int64_t value_width, int64_t index_width, int64_t pool,
        const mfq_tensor_backend::TensorOptions& options,int64_t selected_columns=0);
    // Called after dense weights and predictor load, before expert arenas.
    void finalize(std::size_t minimum_experts = 0);
    void ensure(int64_t tokens);
    void set_reservation(std::function<void(std::size_t)> callback) { reserve_ = std::move(callback); }
    std::shared_ptr<KvOffloadStore> offload() const { return offload_; }
    bool dynamic() const { return options_.dynamic; }
    std::shared_ptr<KvExpertLoan> expert_loan() const { return expert_loan_; }
    std::size_t growth_reserve() const;
    std::size_t expert_budget_charge() const;
    std::vector<std::pair<std::string,double>> metrics() const;
private:
    KvMemoryOptions options_;
    std::vector<std::weak_ptr<KvMemoryAllocation>> allocations_;
    std::shared_ptr<KvOffloadStore> offload_;
    std::function<void(std::size_t)> reserve_;
    bool ready_ = false;
    bool automatic_ram_ = false;
    std::shared_ptr<KvExpertLoan> expert_loan_;
    std::size_t gpu_budget_=0,ram_required_=0;
};
}
