#pragma once

#include <cstddef>
#include <cuda_runtime.h>

// CUDA 13 folded cudaStreamGetCaptureInfo_v3 and cudaGraphNodeGetDependentNodes_v2
// into the unversioned names and deleted the _v2 symbols, so the name to call
// depends on the toolkit major version.

inline cudaError_t bench_stream_captured_graph(cudaStream_t stream,
                                               cudaStreamCaptureStatus *status,
                                               unsigned long long *id,
                                               cudaGraph_t *graph) {
#if CUDART_VERSION >= 13000
    return cudaStreamGetCaptureInfo(stream, status, id, graph, nullptr, nullptr, nullptr);
#elif CUDART_VERSION >= 12030
    return cudaStreamGetCaptureInfo_v3(stream, status, id, graph, nullptr, nullptr, nullptr);
#else
    return cudaStreamGetCaptureInfo_v2(stream, status, id, graph, nullptr, nullptr);
#endif
}

inline cudaError_t bench_dependent_nodes(cudaGraphNode_t node,
                                         cudaGraphNode_t *nodes,
                                         std::size_t *num) {
#if CUDART_VERSION >= 13000
    return cudaGraphNodeGetDependentNodes(node, nodes, nullptr, num);
#elif CUDART_VERSION >= 12030
    return cudaGraphNodeGetDependentNodes_v2(node, nodes, nullptr, num);
#else
    return cudaGraphNodeGetDependentNodes(node, nodes, num);
#endif
}
