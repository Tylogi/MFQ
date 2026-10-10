#pragma once

namespace mfq::cuda {
int mfe_ffn_down_launch_warps(const void* kernel,int routes);
}
