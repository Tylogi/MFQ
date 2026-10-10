#pragma once

// Adapted from Strata's src/kernels/cpu/pool_affinity_win.hpp.
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// MIT license: ../third_party/strata.LICENSE
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstddef>
#include <vector>

namespace mfq::cpu::detail {
inline bool thread_cpu_sets(std::vector<unsigned long>& ids) {
    ULONG count=0;
    if(!GetThreadSelectedCpuSets(GetCurrentThread(),nullptr,0,&count) &&
            GetLastError()!=ERROR_INSUFFICIENT_BUFFER)return false;
    ids.resize(count);
    if(count && !GetThreadSelectedCpuSets(GetCurrentThread(),ids.data(),count,&count))return false;
    ids.resize(count);return true;
}
inline bool cpu_set_for_core(int core,ULONG& id) {
    ULONG bytes=0;
    if(!GetSystemCpuSetInformation(nullptr,0,&bytes,GetCurrentProcess(),0) &&
            GetLastError()!=ERROR_INSUFFICIENT_BUFFER)return false;
    std::vector<unsigned char> buffer(bytes);
    if(bytes && !GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()),
            bytes,&bytes,GetCurrentProcess(),0))return false;
    constexpr std::size_t header=offsetof(SYSTEM_CPU_SET_INFORMATION,CpuSet);
    for(std::size_t offset=0;offset+header<=bytes;) {
        const auto* info=reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data()+offset);
        if(info->Size<header || info->Size>bytes-offset)break;
        if(info->Type==CpuSetInformation && info->Size>=sizeof(SYSTEM_CPU_SET_INFORMATION) &&
                info->CpuSet.Group==core/64 && info->CpuSet.LogicalProcessorIndex==(core&63) &&
                (!info->CpuSet.Allocated || info->CpuSet.AllocatedToTargetProcess)) {
            id=info->CpuSet.Id;return true;
        }
        offset+=info->Size;
    }
    return false;
}
}
#endif
