#pragma once
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace mfq::cuda::internal {
inline void finish_cuda_activity_trace() noexcept {
    if(!std::getenv("MFQ_CUDA_ACTIVITY_PREFIX"))return;
#ifdef _WIN32
    const auto module=GetModuleHandleW(L"cuda_activity_collector.dll");
    const auto finish=module?reinterpret_cast<int(*)()>(GetProcAddress(module,"MFQCompleteCudaActivityTrace")):nullptr;
    const int status=finish?finish():0;
    (void)std::fprintf(stderr,"native_cuda_activity_export status=%d\n",status);
#endif
}
}
