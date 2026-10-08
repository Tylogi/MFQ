#pragma once
#include "moe.h"
#include "mfe_ffn.h"
#include <array>

class MfeFfnRuntime {
    mfq::cuda::MfeFfnBatch batch_{};
    mfq_tensor_backend::Tensor descriptors_,cohorts_,quantization_,sigmoid_,input_,ids_,weights_,shared_gate_;
    mfq_tensor_backend::Tensor hidden_,quantized_,scales_,completion_,output_,kinds_,cpu_;
    std::vector<mfq_tensor_backend::Tensor> owners_;
    std::vector<mfq::cuda::MfePackedProjection> host_views_;
    std::vector<int32_t> host_cohorts_;
    mfq_tensor_backend::Tensor transferred_,transfer_indices_,aborted_,resident_pairs_;
    int quantization_count_=0,quantization_groups_=0;
public:
    MfeFfnRuntime(const std::array<MixedMoeRuntime*,3>& projections,
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& ids,
        const mfq_tensor_backend::Tensor& weights,const mfq_tensor_backend::Tensor& sigmoid,
        const std::array<const NintWeight*,3>& shared={},const mfq_tensor_backend::Tensor& shared_gate={},
        int output_columns=0,int intermediate_columns=0,
        const std::array<std::vector<int32_t>,3>* cohort_by_expert=nullptr,bool output_pairs=false);
    void cpu_results(const mfq_tensor_backend::Tensor& kinds,const mfq_tensor_backend::Tensor& pairs,
        const uint32_t* ready=nullptr);
    void asynchronous(const int32_t* kinds,const uint32_t* plan,const uint32_t* transfer,
        const uint32_t* cpu_ready,const uint32_t* abort,const int32_t* indices,const void* cpu_pairs);
    mfq::cuda::MfePackedProjection expert_view(int projection,int expert,
        const std::vector<int64_t>& field_bytes,int resident_slot)const;
    const mfq_tensor_backend::Tensor& transfer_buffer()const{return transferred_;}
    void prepare();
    void gate_up();
    void resident();
    void wait_transfer();
    void transferred();
    void wait_cpu();
    void down_reduce_compute();
    void down_reduce();
    void run(){prepare();gate_up();down_reduce();}
    const mfq_tensor_backend::Tensor& output()const{return output_;}
    const mfq_tensor_backend::Tensor& hidden()const{return hidden_;}
    const mfq::cuda::MfeFfnBatch& batch()const{return batch_;}
};
