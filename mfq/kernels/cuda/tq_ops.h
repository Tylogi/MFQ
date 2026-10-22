#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#ifdef _WIN32
#define MFQ_TRELLIS_API __declspec(dllexport)
#else
#define MFQ_TRELLIS_API
#endif

// Ordinary 2D projection view. No routing, expert axes or cache ownership.
// All row descriptors select their own 1..8-bit precision and packed offset.
struct TrellisProjectionView {
    const uint32_t* words;
    const uint64_t* descriptors;
    const void* scales;
    const void* relative;
    const float* bias;
    int rows;
    int stored_columns;  // Independent 256-element encoded rings, padded at tail.
    int input_columns;   // Logical input width; padding is masked in the kernel.
    int gslog;
    int columns;         // 0: subgroup selectors; 1: FP16 input-column scales.
};

// dtype=0: FP16 input/output; dtype=1: BF16 input/output. Reconstruction is FP16,
// with FP32 accumulation. Workspaces and outputs are supplied by the caller.
extern "C" MFQ_TRELLIS_API int mfq_tq_gemv(TrellisProjectionView weight,const void* x,void* output,
    int dtype,int warps,int rows_per_block,cudaStream_t stream);
extern "C" MFQ_TRELLIS_API int mfq_tq_swiglu(TrellisProjectionView gate,TrellisProjectionView up,
    const void* x,void* hidden,int dtype,int warps,int rows_per_block,cudaStream_t stream);
extern "C" MFQ_TRELLIS_API int mfq_tq_ffn(TrellisProjectionView gate,TrellisProjectionView up,
    TrellisProjectionView down,const void* x,void* hidden,void* output,int dtype,cudaStream_t stream);
