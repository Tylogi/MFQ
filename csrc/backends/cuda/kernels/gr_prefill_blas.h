#pragma once
#include "native/nint_weight.h"
bool gr_prefill_blas_selected(const NintWeight&,const mfq_tensor_backend::Tensor&);
mfq_tensor_backend::Tensor gr_prefill_blas_fp32(const NintWeight&,const mfq_tensor_backend::Tensor&);
mfq_tensor_backend::Tensor gr_prefill_blas_mix(const NintWeight&,const mfq_tensor_backend::Tensor&,
    const mfq_tensor_backend::Tensor& normalized,int64_t streams);
void prepare_gr_prefill_blas_runtime();
