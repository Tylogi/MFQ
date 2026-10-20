#pragma once
#include "kernels/mfe/ffn.h"

namespace mfq::cuda {
bool mfe_ffn_compact_requested(const MfeFfnBatch&,bool down);
// Return false when any launch policy falls outside the measured contract.
bool mfe_ffn_compact_gate_up(const MfeFfnBatch&,int partition,int width,cudaStream_t);
bool mfe_ffn_compact_down(const MfeFfnBatch&,int partition,cudaStream_t);
bool mfe_ffn_compact_cached_down(const MfeFfnBatch&,const uint32_t* cached,cudaStream_t);
bool mfe_ffn_compact_token_single_gate_up(const MfeFfnBatch&,int partition,int width,cudaStream_t);
bool mfe_ffn_compact_token_single_down(const MfeFfnBatch&,int partition,cudaStream_t);
bool mfe_ffn_compact_group_gate_up(const MfeFfnBatch&,int,cudaStream_t);
}
