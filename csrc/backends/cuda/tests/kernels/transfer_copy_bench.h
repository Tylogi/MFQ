#pragma once
#include "kernels/moe/pipeline.h"

static void benchmark_transfer_copy(bool timings=true) {
    namespace tb=mfq_tensor_backend;
    auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    int blocks=0,threads=0;MFQ_CUDA_CHECK(moe_wire_launch_geometry(&blocks,&threads));
    const auto gpu=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    HostBuffer flags(128,true);uint32_t* flag_device=nullptr;
    MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&flag_device),flags.data(),0));
    auto* flag_host=static_cast<uint32_t*>(flags.data());flag_host[0]=1;flag_host[16]=0;
    int cases=0;
    for(int fields:{0,1,8,64,257})for(int width:{113,4109,131072})for(int misalign:{0,1,4}) {
        const int pitch=(width+31)&~15,bytes=fields*pitch+16;
        const int header=(int(32+fields*24)+15)&~15;
        auto host=tb::zeros({header+bytes},gpu.device(tb::kCPU));auto* raw=host.data_ptr<uint8_t>();
        auto destination=tb::empty({bytes*2},gpu);auto cancel=tb::zeros({1},gpu.dtype(tb::kInt32));
        auto* words=reinterpret_cast<uint64_t*>(raw);
        words[0]=fields;words[1]=header;words[2]=header+fields*pitch;words[3]=header+(fields/2)*pitch;
        for(int i=0;i<bytes;++i)raw[header+i]=uint8_t(i%251);
        for(int i=0;i<fields;++i) {
            words[4+i*3]=reinterpret_cast<uint64_t>(destination.data_ptr<uint8_t>()+i*pitch+misalign);
            words[5+i*3]=header+i*pitch;words[6+i*3]=width;
        }
        auto wire=host.to(tb::kCUDA);
        const auto half=std::size_t(32+fields*sizeof(MoeMappedCopyDescriptor));
        HostBuffer table(half*2,true);void* alias=nullptr;
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(&alias,table.data(),0));
        std::memset(table.data(),0,half*2);
        for(int phase=0;phase<2;++phase) {
            const int first=phase?fields/2:0,last=phase?fields:fields/2;
            auto* h=reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(table.data())+phase*half);
            h[0]=last-first;h[1]=(last-first)*pitch;
            auto* desc=reinterpret_cast<MoeMappedCopyDescriptor*>(h+4);
            for(int i=first;i<last;++i)desc[i-first]={
                reinterpret_cast<uint64_t>(destination.data_ptr<uint8_t>()+bytes+i*pitch+misalign),
                reinterpret_cast<uint64_t>(wire.data_ptr<uint8_t>()+header+i*pitch),
                uint64_t(i-first)*pitch,uint64_t(width)};
        }
        auto snapshot=tb::empty({int64_t(half*2)},gpu);
        for(int fused:{0,1,2,0}) {
            const auto body=[&] {
                for(int phase=0;phase<2;++phase) {
                    const auto* source=static_cast<uint8_t*>(alias)+phase*half;
                    auto* target=snapshot.data_ptr<uint8_t>()+phase*half;
                    if(fused==2)wait_mapped_plan_snapshot(flag_device,flag_device+16,
                        reinterpret_cast<uint32_t*>(cancel.data_ptr()),source,target,fields,stream);
                    else wait_mapped_plan(flag_device,nullptr,nullptr,0,flag_device+16,
                        reinterpret_cast<uint32_t*>(cancel.data_ptr()),stream);
                    if(fused)moe_scatter_wire_and_loan_phase(wire.data_ptr(),fused==2?target:source,fields,
                        reinterpret_cast<uint32_t*>(cancel.data_ptr()),blocks,threads,phase+1,stream,fused==2);
                    else {
                        moe_scatter_wire_phase(wire.data_ptr(),reinterpret_cast<uint32_t*>(cancel.data_ptr()),blocks,threads,phase+1,stream);
                        moe_snapshot_mapped(source,reinterpret_cast<uint32_t*>(cancel.data_ptr()),target,fields,stream);
                        moe_copy_mapped(target,reinterpret_cast<uint32_t*>(target)+4,blocks,threads,stream);
                    }
                }
            };
            for(bool aborted:{true,false}) {
                destination.fill_(255);flag_host[16]=aborted;body();
                auto result=destination.cpu();const auto* actual=result.data_ptr<uint8_t>();
                for(int side=0;side<2;++side)for(int i=0;i<bytes;++i) {
                    const int row=i/pitch,within=i%pitch;
                    const bool copied=!aborted && row<fields && within>=misalign && within<misalign+width;
                    check(actual[side*bytes+i]==(copied?raw[header+row*pitch+within-misalign]:255),
                        "fused wire/loan copy changed data, phase, or padding");
                }
                ++cases;
            }
            if(fused==2) {
                Graph graph;graph.capture_begin_shared();body();graph.capture_end();
                // The wire intentionally remains a nonempty preceding packet.
                // Replays must switch between ignoring it and copying it.
                for(bool empty:{true,false,true,false}) {
                    for(int phase=0;phase<2;++phase) {
                        auto* h=reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(table.data())+phase*half);
                        h[0]=empty?~uint64_t(0):uint64_t(phase?fields-fields/2:fields/2);
                    }
                    destination.fill_(255);graph.replay();auto result=destination.cpu();
                    for(int side=0;side<2;++side)for(int i=0;i<bytes;++i) {
                        const int row=i/pitch,within=i%pitch;
                        const bool copied=!empty && row<fields && within>=misalign && within<misalign+width;
                        check(result.data_ptr<uint8_t>()[side*bytes+i]==(copied?raw[header+row*pitch+within-misalign]:255),
                            "empty transfer replay read stale wire or lost the next packet");
                    }
                    ++cases;
                }
            }
            if(!timings || misalign || width!=131072)continue;
            Graph graph;graph.capture_begin_shared();for(int i=0;i<16;++i)body();graph.capture_end();
            for(int i=0;i<8;++i)graph.replay();
            cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
            MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));for(int i=0;i<64;++i)graph.replay();
            MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
            std::cout<<"transfer_copy fields="<<fields<<" width="<<width<<" fused="<<fused
                <<" both_phases_us="<<ms*1000/(64*16)<<std::endl;
            // Replay sees a new mapped packet and cancellation state.
            flag_host[16]=1;destination.fill_(255);graph.replay();
            auto skipped=destination.cpu();for(int64_t i=0;i<skipped.numel();++i)
                check(skipped.data_ptr<uint8_t>()[i]==255,"cancelled fused transfer graph wrote data");
            flag_host[16]=0;
        }
    }
    context->end_graph_pool(stream);
    std::cout<<"transfer_copy fields, phases, tails, cancellation PASS cases="<<cases<<std::endl;
}
