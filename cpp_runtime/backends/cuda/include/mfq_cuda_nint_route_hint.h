#pragma once
#include "mfq_tensor_backend.h"

// Internal view of a validated, fixed-capacity expert pool. The pointer and
// geometry values share the lifetime of the captured pool/workspace tensors.
struct NintSingleRouteWeight {
    int64_t pointers[9]{};
    int32_t geometry[3]{};
    int32_t local_experts=0;
    int64_t row_metadata=0;
};

bool nint_late_scale_enabled(int64_t input_width,int bits,int device);

bool nint_try_single_route_cuda(
    NintSingleRouteWeight weight,mfq_tensor_backend::Tensor expert_local,
    mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor output,
    bool routed_input,int bits_hint,bool metadata_ready=false);

// Hints select code generation only. Device row metadata is checked on every
// replay, so mixed widths and mutable expert slots retain the general path.
void nint_moe_grouped_matmul_hinted_cuda(
    mfq_tensor_backend::Tensor pointers,mfq_tensor_backend::Tensor params,
    mfq_tensor_backend::Tensor expert_pool,mfq_tensor_backend::Tensor expert_local,
    mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor output,
    int64_t input_width,bool routed_input,int group_hint,int bits_hint);
