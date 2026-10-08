#include "mfe_ffn_launch.h"
#include <cuda_runtime_api.h>
#include <array>
#include <cstdlib>
#include <stdexcept>

namespace mfq::cuda {
int mfe_ffn_down_launch_warps(const void* kernel,int routes) {
    const auto* setting=std::getenv("MFQ_MFE_DOWN_PACK_ROUTES");
    if(!setting || std::atoi(setting)!=1)return routes;
    if(routes<1 || routes>32)throw std::invalid_argument("MFE Down route count exceeds shared result geometry");
    const auto check=[](cudaError_t result) {
        if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
    };
    thread_local int cached_device=-1;
    thread_local const void* cached_kernel=nullptr;
    thread_local std::array<int,33> cached{};
    int device=0;check(cudaGetDevice(&device));
    if(device!=cached_device || kernel!=cached_kernel) {
        cached.fill(0);cached_device=device;cached_kernel=kernel;
    }
    if(cached[routes])return cached[routes];
    int best=routes,best_blocks=0,best_rounds=1;
    // A smaller team repeats routes. Rank completed route groups per SM
    // per loop round, accounting for the final partially used round.
    // Descending candidates keep the largest team when scores tie.
    for(int warps=routes;warps>=1;--warps) {
        int blocks=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks,kernel,32*warps,0));
        const int rounds=(routes+warps-1)/warps;
        if(blocks*best_rounds>best_blocks*rounds) {
            best=warps;best_blocks=blocks;best_rounds=rounds;
        }
    }
    cached[routes]=best;return best;
}
}
