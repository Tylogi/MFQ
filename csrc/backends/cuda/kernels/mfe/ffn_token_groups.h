#pragma once
#include "kernels/mfe/ffn.h"
namespace mfq::cuda {
// Encoded capacities retain pairs for routed experts and use the remaining
// value for shared experts; the existing batch descriptor stays unchanged.
inline constexpr int kMfeSharedGroupEncoding=16;
void mfe_ffn_token_group_shared_plan(const MfeFfnBatch&,cudaStream_t);
void mfe_ffn_quantize_hidden(const MfeFfnBatch&,int partition,cudaStream_t);
bool mfe_ffn_token_group_gate_up(const MfeFfnBatch&,int partition,int width,cudaStream_t);
bool mfe_ffn_token_group_down(const MfeFfnBatch&,int partition,cudaStream_t);
}
