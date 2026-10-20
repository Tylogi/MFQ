#pragma once
#include "native/env.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>

namespace mfq::cuda::dispatch_trace {
inline bool enabled() {
    const auto* option=mfq::cuda::env_lookup("MFQ_TRACE_KERNEL_DISPATCH");
    return option && option[0]=='1';
}
// Inspect the kernel node actually appended by this launch. Diagnostics run
// only during capture; replay gets no tracing kernels or synchronization.
inline void record(cudaStream_t stream,const char* label,const void* expected,const std::string& details) {
    if(!enabled())return;
    cudaStreamCaptureStatus status{};unsigned long long id=0;cudaGraph_t graph=nullptr;
    const cudaGraphNode_t* dependencies=nullptr;size_t count=0;
    // CUDA 13 folded the versioned capture queries into the unversioned API
    // and added an edge-data output before the dependency count.
#if CUDART_VERSION >= 13000
    const auto capture_result=cudaStreamGetCaptureInfo(stream,&status,&id,&graph,&dependencies,nullptr,&count);
#else
    const auto capture_result=cudaStreamGetCaptureInfo_v2(stream,&status,&id,&graph,&dependencies,&count);
#endif
    if(capture_result!=cudaSuccess ||
        status!=cudaStreamCaptureStatusActive || count!=1)return;
    cudaGraphNodeType kind{};
    if(cudaGraphNodeGetType(dependencies[0],&kind)!=cudaSuccess || kind!=cudaGraphNodeTypeKernel)return;
    cudaKernelNodeParams actual{};
    if(cudaGraphKernelNodeGetParams(dependencies[0],&actual)!=cudaSuccess)return;
    static std::mutex mutex;
    static std::unordered_set<std::string> seen;
    const auto key=std::string(label)+details+std::to_string(reinterpret_cast<uintptr_t>(actual.func));
    std::lock_guard<std::mutex> lock(mutex);
    if(!seen.insert(key).second)return;
    cudaFuncAttributes attributes{};
    const auto code=cudaFuncGetAttributes(&attributes,actual.func);
    std::fprintf(stderr,"native_kernel_dispatch name=%s actual_match=%d capture_id=%llu function=%p "
        "threads=%u registers=%d local_bytes=%zu resource_status=%d %s\n",label,
        static_cast<const void*>(actual.func)==expected,id,actual.func,actual.blockDim.x*actual.blockDim.y,
        attributes.numRegs,attributes.localSizeBytes,int(code),details.c_str());
}
}
