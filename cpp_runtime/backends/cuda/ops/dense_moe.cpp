#include "dense_moe.h"
#include "moe.h"
#include "mfq_cuda_moe_ops.h"

#include <algorithm>
#include <climits>
#include <utility>
#include <vector>

namespace {
using Tensor = mfq_tensor_backend::Tensor;

void check_projection(const Tensor& bank, const Tensor& input, const Tensor& ids) {
    MFQ_RUNTIME_CHECK(bank.is_cuda() && bank.is_contiguous() && bank.dim() == 3,
                      "dense routed expert bank must be contiguous CUDA [experts,out,in]");
    MFQ_RUNTIME_CHECK(ids.is_cuda() && ids.dim() == 2,
                      "dense routed ids must be [tokens,routes]");
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.device() == bank.device() && ids.device() == bank.device(),
                      "dense routed tensors must share one CUDA device");
    MFQ_RUNTIME_CHECK(input.dim() == 2 || (input.dim() == 3 && input.size(1) == ids.size(1)),
                      "dense routed activation shape does not match the route plan");
    MFQ_RUNTIME_CHECK(input.size(0) == ids.size(0) && input.size(-1) == bank.size(2),
                      "dense routed input shape does not match the expert bank");
}

bool supports_grouped(const Tensor& bank, const Tensor& ids) {
    static thread_local std::vector<std::pair<int, int>> device_capabilities;
    const int device = bank.get_device();
    auto capability = std::find_if(device_capabilities.begin(), device_capabilities.end(),
        [device](const auto& item) { return item.first == device; });
    if (capability == device_capabilities.end()) {
        int major = 0;
        MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        device_capabilities.emplace_back(device, major);
        capability = device_capabilities.end() - 1;
    }
    const auto dtype = bank.scalar_type();
    return capability->second >= 8 &&
        (dtype == mfq_tensor_backend::kFloat16 || dtype == mfq_tensor_backend::kBFloat16 ||
            dtype == mfq_tensor_backend::kFloat32) &&
        bank.size(0) > 0 && bank.size(0) <= 4096 && bank.size(1) > 0 && bank.size(1) <= INT_MAX - 64 &&
        bank.size(2) > 0 && bank.size(2) <= INT_MAX - 32 && ids.numel() > 0 && ids.numel() <= INT_MAX - 128 &&
        (ids.scalar_type() == mfq_tensor_backend::kInt32 || ids.scalar_type() == mfq_tensor_backend::kInt64);
}
} // namespace

Tensor dense_routed_projection_gather(const Tensor &bank, const Tensor &input, const Tensor &ids,
                               std::size_t gather_budget_bytes) {
    check_projection(bank, input, ids);
    const int64_t rows = ids.size(0), routes = ids.size(1);
    const int64_t output = bank.size(1), width = bank.size(2);
    auto project = [&](const Tensor &chunk_ids, const Tensor &chunk_input) {
        const int64_t count = chunk_ids.size(0);
        auto selected =
            bank.index_select(0, chunk_ids.reshape({-1}).to(mfq_tensor_backend::kInt64))
                .reshape({count, routes, output, width});
        auto source = chunk_input.dim() == 2
                          ? chunk_input.unsqueeze(1).expand({count, routes, width})
                          : chunk_input;
        return mfq_tensor_backend::matmul(selected, source.to(bank.scalar_type()).unsqueeze(-1))
            .squeeze(-1);
    };
    // Every route copies its whole expert matrix, so a long prompt would gather
    // tokens * routes full matrices at once; bound each copy by row chunks.
    const auto row_bytes = static_cast<std::size_t>(routes * output * width) * bank.element_size();
    const auto rows_per_chunk = gather_budget_bytes / std::max<std::size_t>(1, row_bytes);
    const auto chunk_rows = static_cast<int64_t>(std::max<std::size_t>(1, rows_per_chunk));
    if (rows <= chunk_rows)
        return project(ids, input);
    std::vector<Tensor> parts;
    parts.reserve(static_cast<std::size_t>((rows + chunk_rows - 1) / chunk_rows));
    for (int64_t start = 0; start < rows; start += chunk_rows) {
        const auto count = std::min(chunk_rows, rows - start);
        parts.push_back(project(ids.narrow(0, start, count), input.narrow(0, start, count)));
    }
    return mfq_tensor_backend::cat(parts, 0);
}

Tensor dense_routed_projection(const Tensor& bank, const Tensor& input, const Tensor& ids,
                               std::size_t gather_budget_bytes) {
    check_projection(bank, input, ids);
    if (!supports_grouped(bank, ids))
        return dense_routed_projection_gather(bank, input, ids, gather_budget_bytes);
    const MfqCudaGuard guard(bank.device());
    auto route = build_moe_route_plan(ids.to(mfq_tensor_backend::kInt32).contiguous(), int(bank.size(0)));
    return dense_routed_projection(bank, input, route, gather_budget_bytes);
}

Tensor dense_routed_projection(const Tensor& bank, const Tensor& input, const MoeRoutePlan& route,
                               std::size_t gather_budget_bytes) {
    check_projection(bank, input, route.ids);
    if (!supports_grouped(bank, route.ids))
        return dense_routed_projection_gather(bank, input, route.ids, gather_budget_bytes);
    MFQ_RUNTIME_CHECK(route.n_experts == bank.size(0) && route.map_ready,
                      "dense routed projection requires a matching ready route plan");
    const MfqCudaGuard guard(bank.device());
    return dense_moe_grouped_matmul_cuda(bank, input.to(bank.scalar_type()).contiguous(), route.ids,
        route.ids_dst, route.expert_bounds, route.mma_tile_bounds, route.mma_tile_experts, route.mma_tile_m);
}
