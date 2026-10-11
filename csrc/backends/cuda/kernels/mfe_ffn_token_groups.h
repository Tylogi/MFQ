#pragma once
#include "mfe_ffn.h"
namespace mfq::cuda {
bool mfe_ffn_token_group_gate_up(const MfeFfnBatch&,int partition,int width,cudaStream_t);
bool mfe_ffn_token_group_down(const MfeFfnBatch&,int partition,cudaStream_t);
}
