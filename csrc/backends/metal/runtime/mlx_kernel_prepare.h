#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <mlx/mlx.h>
#include "mlx_sampling.h"

namespace mfq::metal {

class MlxPreparableKernel {
public:
    virtual ~MlxPreparableKernel() = default;
    virtual std::string preparation_key() const = 0;
    virtual void prepare_gpu() = 0;
};

class MlxKernelPreparation {
public:
    MlxKernelPreparation(int context, int chunk_size);
    ~MlxKernelPreparation();
    MlxKernelPreparation(const MlxKernelPreparation&) = delete;
    MlxKernelPreparation& operator=(const MlxKernelPreparation&) = delete;
    static MlxKernelPreparation* current() noexcept;
    void collect(const std::function<void()>& build);
    void add(const mlx::core::array& output);
    void finish();
    void collect_sampling(int vocab, const MlxSamplingParams& defaults);
    const std::vector<int>& row_buckets() const noexcept { return rows_; }
    int context() const noexcept { return context_; }
    void set_routes(int routes) noexcept { routes_ = routes; }
    int routes() const noexcept { return routes_; }
    std::size_t size() const noexcept { return jobs_.size(); }
    const std::unordered_set<std::string>& unsupported() const noexcept { return unsupported_; }

private:
    static thread_local MlxKernelPreparation* active_;
    MlxKernelPreparation* previous_;
    bool collecting_ = false;
    int routes_ = 0;
    int context_ = 0;
    std::vector<int> rows_;
    std::unordered_set<std::string> keys_;
    std::unordered_set<std::string> unsupported_;
    std::vector<std::function<void()>> jobs_;
};

} // namespace mfq::metal
