#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

struct E8DenseProbeWeight {
    const uint8_t *indices=nullptr,*signs=nullptr,*states=nullptr,*dense=nullptr;
    const float* anchors=nullptr;
    const int8_t* codebook=nullptr;
    int64_t index_bytes=0,sign_bytes=0,dense_bytes=0;
    int rows=0,vectors=0,groups=0,sign_mode=0;
    uint32_t bank_map=0;
};
void e8_dense_probe_launch(int format,bool dense,int lanes,const E8DenseProbeWeight* weights,
    int experts,int rows,const int8_t* input,const float* scales,float* output,cudaStream_t stream,
    int decoder=0);
void e8_dense_probe_validate(int format,const E8DenseProbeWeight* weights,int experts,int rows,
    uint32_t* errors,cudaStream_t stream);
cudaFuncAttributes e8_dense_probe_attributes(int format,bool dense,int lanes,int decoder=0);
void d4_dense_probe_launch(int format,bool dense,int lanes,const E8DenseProbeWeight* weights,
    int experts,int rows,const int8_t* input,const float* scales,float* output,cudaStream_t stream,bool narrow=false);
void d4_dense_probe_validate(int format,const E8DenseProbeWeight* weights,int experts,int rows,
    uint32_t* errors,cudaStream_t stream);
cudaFuncAttributes d4_dense_probe_attributes(int format,bool dense,int lanes,bool narrow=false);
