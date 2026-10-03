#pragma once

#include <torch/extension.h>
#include <algorithm>
#include <vector>

// Larger submissions have caused CUDA device failures on the offline NVQ
// search path. Keep complete rows together (anchor refits reduce over a row).
inline int64_t nvq_group_step(int64_t groups_per_row, int64_t requested) {
    constexpr int64_t max_groups = 4096;
    TORCH_CHECK(groups_per_row > 0 && groups_per_row <= max_groups,
                "NVQ CUDA search requires 1..4096 groups per row");
    TORCH_CHECK(requested > 0, "NVQ group_chunk must be positive");
    const auto limit = std::min(requested, max_groups);
    return std::max<int64_t>(1, limit / groups_per_row) * groups_per_row;
}

template <typename Run>
std::vector<torch::Tensor> nvq_chunked(int64_t count, int64_t step, Run run) {
    TORCH_CHECK(count > 0, "NVQ CUDA search requires nonempty input");
    if (count <= step) return run(0, count);
    std::vector<torch::Tensor> output;
    for (int64_t begin = 0; begin < count; begin += step) {
        const auto length = std::min(step, count - begin);
        const auto part = run(begin, length);
        if (output.empty()) {
            for (const auto& value : part) {
                auto shape = value.sizes().vec();
                shape[0] = count;
                output.push_back(torch::empty(shape, value.options()));
            }
        }
        for (size_t i = 0; i < output.size(); ++i) {
            output[i].narrow(0, begin, length).copy_(part[i]);
        }
    }
    return output;
}
