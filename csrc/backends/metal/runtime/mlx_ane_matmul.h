#pragma once

#include "mfq_ane.h"
#include <mlx/mlx.h>
#include <map>
#include <mutex>
#include <tuple>

namespace mfq::metal {

class MlxAneMatmul : public std::enable_shared_from_this<MlxAneMatmul> {
public:
    mlx::core::Stream cpu() const;
    mlx::core::array operator()(const mlx::core::array&, const mlx::core::array&);
    void prepare(int rows, int inner, int columns);
    std::vector<float> evaluate(int rows, int inner, int columns,
        std::span<const float>, std::span<const float>);
    std::vector<float> evaluate(const mlx::core::array&, const mlx::core::array&);
    std::size_t bytes() const;
private:
    std::shared_ptr<AneMatmul> kernel(int rows, int inner, int columns);
    std::vector<float> evaluate_strided(int rows, int inner, int columns,
        const float*, const float*, std::size_t, std::size_t, std::size_t, std::size_t);
    mutable std::mutex mutex_;
    std::map<std::tuple<int, int, int>, std::shared_ptr<AneMatmul>> kernels_;
};

} // namespace mfq::metal
