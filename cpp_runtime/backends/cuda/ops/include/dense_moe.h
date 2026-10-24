#pragma once

#include "mfq_tensor_backend.h"
#include <cstddef>

struct MoeRoutePlan;

inline constexpr std::size_t kDenseRoutedGatherBudgetBytes = std::size_t(512) << 20;

// Apply [experts,out,in] to [tokens,in] or [tokens,routes,in], returning
// [tokens,routes,out] in bank precision with FP32 accumulation. Ready plans
// are reused across Up, Gate and Down. No expert matrices are copied.
// Pre-sm_80 devices, F64, >4096 experts, empty dimensions and dimensions beyond
// the int32 kernel ABI retain the chunked gather fallback. IDs must identify valid experts.
// gather_budget_bytes bounds only the fallback's copies of expert matrices.
mfq_tensor_backend::Tensor dense_routed_projection(
    const mfq_tensor_backend::Tensor& bank, const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& ids,
    std::size_t gather_budget_bytes = kDenseRoutedGatherBudgetBytes);
mfq_tensor_backend::Tensor dense_routed_projection(
    const mfq_tensor_backend::Tensor& bank, const mfq_tensor_backend::Tensor& input,
    const MoeRoutePlan& route,
    std::size_t gather_budget_bytes = kDenseRoutedGatherBudgetBytes);

// Bounded legacy implementation, also used as the benchmark/reference path.
mfq_tensor_backend::Tensor dense_routed_projection_gather(
    const mfq_tensor_backend::Tensor& bank, const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& ids,
    std::size_t gather_budget_bytes = kDenseRoutedGatherBudgetBytes);
