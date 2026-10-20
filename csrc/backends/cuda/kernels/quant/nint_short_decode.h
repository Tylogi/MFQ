#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>

struct NintWholeArgs {
    const uint8_t *weights,*bits,*scale,*minimum;
    const int64_t* offsets;
    const float *outer_scale,*outer_minimum;
    const int8_t* activation;
    const float* activation_scale;
    __half* output;
    uint64_t bytes;
    int rows,groups;
    int activation_rows=1;
    bool group_load=false;
    const int* activation_sum=nullptr;
    const int* activation_interleaved=nullptr;
};

// Called only after the canonical packed-decode eligibility checks.
bool nint_launch_fixed_decode(NintWholeArgs args,int group_size,int uniform_bits,cudaStream_t stream);

bool nint_default_activation_sum(int group_size,int groups,int rows,int uniform_bits,bool interleaved);

void nint_launch_grouped_q8(NintWholeArgs args,cudaStream_t stream);
