#pragma once
#include "packed_nint.h"
#include "nvq_group.h"
#include "nint.h"
#include "vq.h"
#include <functional>

// The plan borrows an immutable expert lease. It never converts activations
// to integer codes, expands weights, or launches another thread pool.
struct CpuProjectionRows {
    int width=0, outputs=0;
    mfq::cpu::PackedNintView nint;
    mfq::cpu::NvqDecodeView nvq{};
    mfq::cpu::PackedNintRows nint_rows=nullptr;
    mfq::cpu::NvqRowsDot nvq_rows=nullptr;
    std::function<void(const float*,int,int,float*,int,int,int)> scalar_rows;
    void run(const float* input,int batch,int input_stride,float* output,
        int output_stride,int begin,int end) const {
        if(nint_rows) nint_rows(nint,input,batch,input_stride,output,output_stride,begin,end);
        else if(nvq_rows) nvq_rows(nvq,input,batch,input_stride,output,output_stride,begin,end);
        else scalar_rows(input,batch,input_stride,output,output_stride,begin,end);
    }
};
CpuProjectionRows make_cpu_projection_rows(const NintWeight&);
CpuProjectionRows make_cpu_projection_rows(const NvqWeight&);
