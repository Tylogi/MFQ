#pragma once

namespace mfq::cuda {
struct MfeFfnBatch;
bool mfe_ffn_e8_narrow_requested(const MfeFfnBatch&,bool down);
int mfe_ffn_down_launch_warps(const void* kernel,int routes,int tokens);
bool mfe_ffn_format_specialization_enabled();
bool mfe_ffn_nint_whole_requested();
}
