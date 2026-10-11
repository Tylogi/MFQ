#pragma once
#include "packed_nvq.cuh"

namespace mfq::cuda {
struct NvqRoutedReuseParams {
    packed_nvq::NvqDeviceWeight weight{};
    const int64_t* pointers=nullptr;
    const int64_t* sizes=nullptr;
    const int32_t* params=nullptr;
    const int32_t* expert_pool=nullptr;
    const int32_t* expert_local=nullptr;
    const int32_t* ids=nullptr;
    const int32_t* active=nullptr;
    const int8_t* input=nullptr;
    const float* scales=nullptr;
    __half* output=nullptr;
    int format=0,experts=0,local_experts=0,pools=0,rows=0,routes=0,pairs=0;
    bool down=false;
};
bool nvq_launch_routed_reuse(const NvqRoutedReuseParams&,cudaStream_t);
}
