#pragma once

#include "rope.h"

namespace mfq::cuda {

// Values and validity travel together. A missing snapshot starts with no valid groups.
struct PooledCacheState {
    mfq_tensor_backend::Tensor values;
    int64_t completed = 0;
};

// Request-owned completed-group watermark, shared by eager, offload and graph updates.
class PooledCache {
public:
    using Tensor = mfq_tensor_backend::Tensor;
    int64_t completed() const { return completed_; }
    int64_t processed_groups() const { return processed_; }
    const Tensor& storage() const { return values_; }
    const Tensor& positions() const { return positions_; }
    std::size_t growth_bytes(int64_t tokens,int64_t pool,int64_t width,std::size_t element_size) const;
    std::size_t position_bytes(int64_t capacity) const;
    // Optional device instrumentation counts rows that actually enter the fused kernel.
    void count_group_rows(const Tensor& counter) { processed_rows_=counter; }
    void set_incremental(bool enabled) { incremental_=enabled; }
    void reset();
    void truncate(int64_t tokens,int64_t pool);
    void bind(const Tensor& values);
    void reserve(const Tensor& raw,int64_t capacity);
    Tensor update(const RotaryEmbedding& rotary,const Tensor& raw,const Tensor& weight,
        const Tensor& full_positions,int64_t tokens,int64_t pool,double eps,bool cache = true);
    // Offloaded source ranges may begin after zero and write to a staging tensor.
    Tensor update_range(const RotaryEmbedding& rotary,const Tensor& raw,const Tensor& weight,
        const Tensor& full_positions,int64_t first,int64_t count,int64_t pool,double eps);
    void prepare_graph(const Tensor& full_positions,int64_t capacity);
    void update_graph(const RotaryEmbedding& rotary,const Tensor& raw,const Tensor& weight,
        const Tensor& positions,const Tensor& cache_positions,int64_t pool,double eps);
    void advance_graph(int64_t tokens,int64_t pool);
    PooledCacheState capture(int64_t tokens,int64_t pool) const;
    void restore(const PooledCacheState& state,int64_t tokens,int64_t pool);
private:
    Tensor values_,positions_,processed_rows_;
    int64_t completed_ = 0,processed_ = 0;
    bool incremental_ = true;
};

} // namespace mfq::cuda
