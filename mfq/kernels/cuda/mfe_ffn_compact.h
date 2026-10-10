#pragma once
#include "mfe_ffn.h"

namespace mfq::cuda {
bool mfe_ffn_compact_requested(const MfeFfnBatch&,bool down);
// Return false when any launch policy falls outside the measured contract.
bool mfe_ffn_compact_gate_up(const MfeFfnBatch&,int partition,int width,cudaStream_t);
bool mfe_ffn_compact_down(const MfeFfnBatch&,int partition,cudaStream_t);
}
