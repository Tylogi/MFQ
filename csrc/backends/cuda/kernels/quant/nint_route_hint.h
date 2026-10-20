#pragma once
#include "native/tensor/backend.h"
#include <vector>

// A serial group of projections can quantize its common input for all GS
// values in one launch. This state is capture-local, never a replay cache.
struct NintPreparedGroupInput {
    int gs=0;
    mfq_tensor_backend::Tensor codes,scale,sum,interleaved;
};
struct NintInputQuantizationScope {
    mfq_tensor_backend::Tensor input;
    std::vector<NintPreparedGroupInput> groups;
    NintInputQuantizationScope* previous=nullptr;
    NintInputQuantizationScope(mfq_tensor_backend::Tensor input,const std::vector<int>& group_sizes);
    ~NintInputQuantizationScope();
    NintInputQuantizationScope(const NintInputQuantizationScope&)=delete;
    NintInputQuantizationScope& operator=(const NintInputQuantizationScope&)=delete;
};

// Internal view of a validated, fixed-capacity expert pool. The pointer and
// geometry values share the lifetime of the captured pool/workspace tensors.
struct NintSingleRouteWeight {
    int64_t pointers[9]{};
    int32_t geometry[3]{};
    int32_t local_experts=0;
    int64_t row_metadata=0;
};

bool nint_late_scale_enabled(int64_t input_width,int bits,int device);

mfq_tensor_backend::Tensor nint_matmul_fixed_bits_ws_cuda(
    mfq_tensor_backend::Tensor bitstream,mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,mfq_tensor_backend::Tensor subgroup_scale,
    mfq_tensor_backend::Tensor subgroup_minimum,mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor neuron_minimum,mfq_tensor_backend::Tensor input,
    int64_t group_size,mfq_tensor_backend::Tensor quantized_input,
    mfq_tensor_backend::Tensor input_scale,int uniform_bits);

mfq_tensor_backend::Tensor nint_matmul_input_mul_fixed_bits_ws_cuda(
    mfq_tensor_backend::Tensor bitstream,mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,mfq_tensor_backend::Tensor subgroup_scale,
    mfq_tensor_backend::Tensor subgroup_minimum,mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor neuron_minimum,mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,int64_t activation_mode,int64_t group_size,
    mfq_tensor_backend::Tensor quantized_input,mfq_tensor_backend::Tensor input_scale,int uniform_bits);

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
