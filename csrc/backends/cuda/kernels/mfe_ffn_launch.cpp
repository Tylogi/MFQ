#include "mfe_ffn_launch.h"
#include "mfe_ffn.h"
#include "mfe_ffn_compact.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>

namespace mfq::cuda {
bool mfe_ffn_compact_requested(const MfeFfnBatch& b,bool down) {
    if(!b.compact_nvq || !b.default_math || b.shared_dense_math ||
            (b.nvq_format_mask&~0x3f922u) || b.tokens<1 || b.tokens>8 ||
            b.routes!=10 || b.input_width!=2560 || b.output_width!=2560 ||
            b.intermediate!=640 || (b.shared && b.shared_intermediate!=640))return false;
    // This module independently proves default arithmetic. The legacy
    // DEFAULT_MATH flag selects the fallback kernel's code specialization;
    // COMPACT=0 retains that choice exactly.
    if(const auto* setting=std::getenv("MFQ_MFE_NVQ_COMPACT"))
        return setting[0]=='1' || setting[0]==(down?'3':'2');
    thread_local int cached_device=-1;thread_local bool sm86=false;
    int device=0;
    if(cudaGetDevice(&device)!=cudaSuccess)throw std::runtime_error("MFE compact device query failed");
    if(device!=cached_device) {
        int major=0,minor=0;
        if(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device)!=cudaSuccess ||
           cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device)!=cudaSuccess)
            throw std::runtime_error("MFE compact architecture query failed");
        sm86=major==8 && minor==6;cached_device=device;
    }
    return sm86;
}
bool mfe_ffn_nint_whole_requested() {
    if(const char* setting=std::getenv("MFQ_MFE_NINT_WHOLE"))return setting[0]!='0';
    thread_local int cached_device=-1;
    thread_local bool sm86=false;
    int device=0;
    if(cudaGetDevice(&device)!=cudaSuccess)throw std::runtime_error("MFE NINT device query failed");
    if(device!=cached_device) {
        int major=0,minor=0;
        if(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device)!=cudaSuccess ||
           cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device)!=cudaSuccess)
            throw std::runtime_error("MFE NINT architecture query failed");
        sm86=major==8 && minor==6;cached_device=device;
    }
    return sm86;
}
bool mfe_ffn_e8_narrow_requested(const MfeFfnBatch& b,bool down) {
    if(const char* setting=std::getenv("MFQ_MFE_E8_NARROW"))
        return setting[0]=='1' || setting[0]==(down?'3':'2');
    // Expert-local rebasing permits the large asynchronous cache arenas as
    // well as resident pools. Both paths have exact-output checks; ordinary
    // SM86 OFF/ON/OFF also covers this top-10 geometry. Independent descriptor
    // bounds and arithmetic proofs in the CUDA wrapper remain mandatory.
    if(b.tokens!=1 || b.routes!=10 || b.input_width!=2560 || b.output_width!=2560 ||
            b.intermediate!=640 || (b.shared && b.shared_intermediate!=640))return false;
    thread_local int cached_device=-1;
    thread_local bool sm86=false;
    int device=0;
    if(cudaGetDevice(&device)!=cudaSuccess)throw std::runtime_error("MFE device query failed");
    if(device!=cached_device) {
        int major=0,minor=0;
        if(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device)!=cudaSuccess ||
           cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device)!=cudaSuccess)
            throw std::runtime_error("MFE architecture query failed");
        sm86=major==8 && minor==6;cached_device=device;
    }
    return sm86;
}

bool mfe_ffn_format_specialization_enabled() {
    if(const char* setting=std::getenv("MFQ_MFE_FORMAT_SPECIALIZE"))return setting[0]!='0';
    thread_local int cached_device=-1;thread_local bool sm86=false;
    int device=0;
    if(cudaGetDevice(&device)!=cudaSuccess)throw std::runtime_error("MFE device query failed");
    if(device!=cached_device) {
        int major=0,minor=0;
        if(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device)!=cudaSuccess ||
           cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device)!=cudaSuccess)
            throw std::runtime_error("MFE architecture query failed");
        sm86=major==8 && minor==6;cached_device=device;
    }
    return sm86;
}
int mfe_ffn_down_launch_warps(const void* kernel,int routes,int tokens) {
    if(const auto* requested=std::getenv("MFQ_MFE_DOWN_WARPS")) {
        char* end=nullptr;const auto value=std::strtol(requested,&end,10);
        if(end==requested || *end || value<1 || value>32 || routes<1 || routes>32)
            throw std::invalid_argument("MFE Down warp override must be 1..32 with 1..32 routes");
        return std::min(int(value),routes);
    }
    const auto* setting=std::getenv("MFQ_MFE_DOWN_PACK_ROUTES");
    if(setting && std::atoi(setting)!=1)return routes;
    if(!setting && (tokens<1 || tokens>8 || (routes!=10 && routes!=11)))return routes;
    if(routes<1 || routes>32)throw std::invalid_argument("MFE Down route count exceeds shared result geometry");
    const auto check=[](cudaError_t result) {
        if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
    };
    thread_local int cached_device=-1;
    thread_local bool sm86=false;
    thread_local const void* cached_kernel=nullptr;
    thread_local std::array<int,33> cached{};
    int device=0;check(cudaGetDevice(&device));
    if(device!=cached_device) {
        int major=0,minor=0;
        check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
        check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));
        sm86=major==8 && minor==6;
        cached_device=device;cached_kernel=nullptr;
    }
    // Local SM86 measurements cover top-10 decoding, with/without a shared
    // expert. Other geometries and architectures retain explicit opt-in.
    if(!setting && !sm86)return routes;
    if(kernel!=cached_kernel) {
        cached.fill(0);cached_kernel=kernel;
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
