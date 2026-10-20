#pragma once
#include "moe.h"
#include "kernels/mfe/ffn.h"
#include "kernels/mfe/ffn.h"
#include "kernels/mfe/ffn_prepare.h"
#include "native/context.h"
#include <array>
#include <map>

class MfeFfnRuntime {
    mfq::cuda::MfeFfnBatch batch_{};
    mfq_tensor_backend::Tensor descriptors_,cohorts_,quantization_,sigmoid_,input_,ids_,weights_,shared_gate_;
    mfq_tensor_backend::Tensor bound_shared_gate_;
    mfq_tensor_backend::Tensor hidden_,quantized_,scales_,completion_,output_,kinds_,cpu_;
    std::vector<mfq_tensor_backend::Tensor> owners_;
    std::vector<mfq::cuda::MfePackedProjection> host_views_;
    std::vector<int32_t> host_cohorts_;
    std::map<std::pair<int,int>,std::pair<mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor>> input_quantizers_;
    mfq_tensor_backend::Tensor transferred_,transfer_indices_,aborted_,resident_pairs_;
    mfq_tensor_backend::Tensor token_group_counts_,token_group_slots_,token_group_buckets_;
    int quantization_count_=0,quantization_groups_=0;
    bool dispatch_trace_reported_=false;
    bool combined_gate_up_=false;
    bool split_shared_groups_=false;
    mfq_tensor_backend::Tensor early_shared_descriptors_,early_shared_quantizers_;
    std::vector<mfq_tensor_backend::Tensor> early_shared_inputs_;
    mfq::cuda::MfePrepareSharedGate early_shared_gate_{};
    int early_shared_quantizer_count_=0,early_shared_groups_=0;
    std::unique_ptr<mfq::cuda::Stream> shared_stream_;
    std::unique_ptr<mfq::cuda::Event> shared_fork_,shared_done_;
    bool shared_in_flight_=false,shared_stream_reported_=false;
    bool shared_short_encoding_=false,shared_short_reported_=false;
    bool shared_early_reported_=false;
    mfq::cuda::MfeFfnBatch arithmetic_batch()const {
        auto b=batch_;
        // Keep the shared slot and reduction geometry intact. Only its
        // arithmetic is delegated to shared_stream_.
        if(shared_in_flight_)b.shared_intermediate=0;
        return b;
    }
    void join_shared();
    void start_shared(bool early);
    const int32_t* transfer_route_plan_=nullptr;
    mfq_tensor_backend::Tensor primary_cohorts_,cached_snapshot_tags_;
    mfq_tensor_backend::Tensor cached_transfer_routes_;
    bool cached_transfer_routes_ready_=false;
    mfq_tensor_backend::Tensor prepared_gate_terms_,prepared_gate_completed_;
    mfq::cuda::MfeCachedDispatch cached_dispatch_{};
    void trace_dispatch();
public:
    MfeFfnRuntime(const std::array<MixedMoeRuntime*,3>& projections,
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& ids,
        const mfq_tensor_backend::Tensor& weights,const mfq_tensor_backend::Tensor& sigmoid,
        const std::array<const NintWeight*,3>& shared={},const mfq_tensor_backend::Tensor& shared_gate={},
        int output_columns=0,int intermediate_columns=0,
        const std::array<std::vector<int32_t>,3>* cohort_by_expert=nullptr,bool output_pairs=false,
        bool combined_gate_up=false,int token_group_capacity_hint=0,double swiglu_limit=0);
    mfq::cuda::MfePackedProjection projection_view(int projection,mfq::cuda::MfePackedProjection view)const;
    void cpu_results(const mfq_tensor_backend::Tensor& kinds,const mfq_tensor_backend::Tensor& pairs,
        const uint32_t* ready=nullptr);
    void asynchronous(const int32_t* kinds,const uint32_t* plan,const uint32_t* transfer,
        const uint32_t* cpu_ready,const uint32_t* abort,const int32_t* indices,const void* cpu_pairs);
    void bind_window_inputs(const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& ids,const mfq_tensor_backend::Tensor& weights);
    void use_staged_inputs();
    void bind_shared_gate(const mfq_tensor_backend::Tensor& gate);
    // Caller publishes the complete packet before transfer_ready, and keeps
    // its mapped allocation alive across graph replays.
    void bind_transfer_route_plan(const int32_t* plan){transfer_route_plan_=plan;}
    void enable_cached_dispatch(const mfq::cuda::MfeCachedExpert* mapped,
        const std::array<std::vector<std::vector<int64_t>>,3>& cohort_bytes,uint32_t* snapshot_ready);
    bool try_fuse_prepare_shared_gate(const NintWeight&,const mfq_tensor_backend::Tensor& output);
    bool shared_gate_in_prepare()const{return cached_dispatch_.gate.activation!=nullptr;}
    mfq::cuda::MfePackedProjection expert_view(int projection,int expert,
        const std::vector<int64_t>& field_bytes,int resident_slot)const;
    const mfq_tensor_backend::Tensor& transfer_buffer()const{return transferred_;}
    // Initialization only: shared scalar gates may reuse prepare()'s exact
    // group quantization without a second quantization kernel.
    std::pair<mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor> reserve_input_quantization(int groups,int group_size);
    void prepare(int32_t* published_ids=nullptr,uint32_t* ready=nullptr);
    void gate_up();
    void resident();
    void wait_transfer(const void* mapped_table=nullptr,void* snapshot=nullptr,uint64_t capacity=0,
        const void* mapped_descriptors=nullptr,uint64_t descriptor_capacity_words=0);
    void transferred();
    void wait_cpu();
    void down_reduce_compute();
    void down_reduce();
    void run(){prepare();gate_up();down_reduce();}
    const mfq_tensor_backend::Tensor& output()const{return output_;}
    const mfq_tensor_backend::Tensor& hidden()const{return hidden_;}
    const mfq_tensor_backend::Tensor& cached_transfer_routes()const{return cached_transfer_routes_;}
    const mfq::cuda::MfeFfnBatch& batch()const{return batch_;}
};
