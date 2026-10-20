// CUDA runtime compatibility surface for compiling the backend's CUDA test
// sources directly as HIP translation units. hipcc accepts <<<>>> launches
// natively; only the cuda_* runtime API names need mapping.
#pragma once

#include "hip/hip_runtime.h"

using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaError_t = hipError_t;

#define cudaSuccess hipSuccess
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetLastError hipGetLastError
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventDestroy hipEventDestroy
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess

// Backend entry points renamed mfq_cuda_* -> mfq_hip_* in the HIP tree.
#define mfq_cuda_graph_capture_supported mfq_hip_graph_capture_supported
#define mfq_cuda_empty_cache mfq_hip_empty_cache
#define mfq_cuda_manual_seed_all mfq_hip_manual_seed_all
#define mfq_cuda_memory_stats mfq_hip_memory_stats
#define mfq_cuda_record_stream mfq_hip_record_stream
#define mfq_cuda_synchronize mfq_hip_synchronize
#define mfq_cuda_tensor_view mfq_hip_tensor_view
