#pragma once
#include "kernels/mfe/ffn.h"

namespace mfq::cuda {
struct MfePrepareSharedGate;
// Independent input buffers let shared math start beside route preparation.
void mfe_ffn_shared_prepare_input(const __half*,int,const MfeInputQuantization*,int,int,cudaStream_t);
// Uses the original batch, including the shared route's storage stride.
// Writes raw shared Down results; the caller retains the existing reduction.
void mfe_ffn_shared_resident(const MfeFfnBatch&,cudaStream_t);
// Single-token shared Down also computes the ordered scalar shared gate.
void mfe_ffn_shared_resident_with_gate(const MfeFfnBatch&,const MfePrepareSharedGate&,cudaStream_t);
// All shared projections must have GS24/28/32, with the default arithmetic.
bool mfe_ffn_shared_short_resident(const MfeFfnBatch&,cudaStream_t);
}
