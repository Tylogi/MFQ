#pragma once

#include "core/decode_graph.h"
#include "runtime/batch_decode_inputs.h"
#include "runtime/decode_window.h"

#include <chrono>
#include <map>
#include <set>

namespace mfq::cuda {

// The last bucket may be shorter than a power of two when capacity is not one.
int64_t batch_decode_row_bucket(int64_t rows, int64_t maximum);
int64_t batch_decode_cache_capacity(int64_t required, int64_t rows, int64_t maximum, int64_t initial);

struct BatchDecodeGraphKey {
    int64_t rows = 0;
    int64_t length = 0;
    int64_t capacity = 0;
    std::vector<int64_t> state_capacities;
    auto operator<=>(const BatchDecodeGraphKey&) const = default;
};

// Borrow the owner's participant pools without transferring their lifetime.
class BatchDecodeCaptureScope {
public:
    BatchDecodeCaptureScope(MfqCudaGraph&, const std::vector<MfqCudaStream>&);
    ~BatchDecodeCaptureScope();
    void finish();
private:
    MfqCudaGraph& graph_;
    std::vector<MfqCudaStream> participants_;
    bool finished_ = false;
};

void mask_batch_routes(CudaExecutionContext&, std::vector<mfq_tensor_backend::Tensor>& selected);
// Join uploaded inputs and state commits onto the graph's launch stream.
void join_batch_decode_stream(const MfqCudaStream& worker, const MfqCudaStream& primary);

// All entries execute serially on this owner's streams and borrow one pool.
// No entry owns request state or uses membership as part of its capture key.
class BatchDecodeGraphCache {
public:
    using Tensor = mfq_tensor_backend::Tensor;
    struct Entry {
        BatchDecodeGraphKey key;
        std::unique_ptr<BatchDecodeInputs> inputs;
        std::unique_ptr<MfqCudaGraph> graph;
        std::unique_ptr<DecodeWindow> window;
        Tensor output;
        bool fresh_capture = false;
        bool valid = false;
    };
    explicit BatchDecodeGraphCache(bool current_stream = false);
    ~BatchDecodeGraphCache();
    Entry& get(BatchDecodeGraphKey key, int64_t fields, int device);
    void prepare_pool(const ParallelConfig&, const ModelParallelCollectiveRuntime&, bool external = false);
    void clear();
    bool has_graphs() const;
    void captured(Entry&, double milliseconds);
    void replayed(Entry&);
    template<class Warmup> void prepare_output(int64_t rows, bool greedy, Warmup&& warmup) {
        const auto shape = std::make_pair(rows, greedy);
        if (prepared_outputs_.contains(shape))
            return;
        warmup();
        prepared_outputs_.insert(shape);
    }
    const MfqCudaStream& stream() const { return stream_; }
    const std::vector<MfqCudaStream>& compute_streams() const { return compute_streams_; }
    std::vector<MfqCudaStream> participants(const ModelParallelCollectiveRuntime&) const;
    uint64_t captures = 0, replays = 0, membership_captures = 0, bucket_reuses = 0;
    double capture_ms = 0;
private:
    MfqCudaStream stream_;
    std::vector<MfqCudaStream> compute_streams_;
    std::unique_ptr<MfqCudaGraph> pool_;
    std::map<BatchDecodeGraphKey, std::unique_ptr<Entry>> entries_;
    std::set<std::pair<int64_t, bool>> prepared_outputs_;
    bool external_pool_ = false;
};

// FFN tiles retain the original row offset when applying the device mask.
class BatchDecodeMaskScope {
public:
    BatchDecodeMaskScope(CudaExecutionContext&, mfq_tensor_backend::Tensor mask, int64_t offset = 0);
    ~BatchDecodeMaskScope();
private:
    CudaExecutionContext& execution_;
    mfq_tensor_backend::Tensor previous_;
    int64_t offset_;
};

class MoeContinuousBatchCacheScope {
public:
    MoeContinuousBatchCacheScope(CudaExecutionContext& execution, bool enabled)
        : execution_(execution), previous_(execution.continuous_batch_cache_serial) {
        execution_.continuous_batch_cache_serial = enabled;
    }
    ~MoeContinuousBatchCacheScope() { execution_.continuous_batch_cache_serial = previous_; }
private:
    CudaExecutionContext& execution_;
    bool previous_;
};

} // namespace mfq::cuda
