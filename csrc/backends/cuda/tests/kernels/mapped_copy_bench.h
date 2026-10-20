#pragma once
#include "kernels/moe/pipeline.h"

static void benchmark_mapped_copy(bool timings=true,bool pinned_source=false) {
    namespace tb=mfq_tensor_backend;
    auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    int blocks=0,threads=0;MFQ_CUDA_CHECK(moe_wire_launch_geometry(&blocks,&threads));
    std::cout<<"mapped_copy launch blocks="<<blocks<<" threads="<<threads<<" source="<<(pinned_source?"RAM":"VRAM")<<"\n";
    for(int fields:{0,1,8,64,257})for(int width:{113,4096+13,131072})for(int misalign:{0,1,4}) {
        const int pitch=(width+31)&~15,bytes=fields*pitch+16;
        auto host=tb::empty({bytes},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(pinned_source));
        auto* values=host.data_ptr<uint8_t>();for(int i=0;i<bytes;++i)values[i]=uint8_t(i%251);
        auto source=pinned_source?host:host.to(tb::kCUDA);
        const auto gpu=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
        auto cancel=tb::zeros({1},gpu.dtype(tb::kInt32));
        auto destination=tb::empty({bytes},gpu);
        auto* source_pointer=source.data_ptr<uint8_t>();
        if(pinned_source)MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&source_pointer),host.data_ptr(),0));
        std::vector<int64_t> descriptors(4+fields*4);descriptors[0]=fields;descriptors[1]=int64_t(fields)*pitch;
        for(int i=0;i<fields;++i) {
            descriptors[4+i*4]=int64_t(reinterpret_cast<uintptr_t>(destination.data_ptr<uint8_t>()+i*pitch+misalign));
            descriptors[5+i*4]=int64_t(reinterpret_cast<uintptr_t>(source_pointer+i*pitch));
            descriptors[6+i*4]=int64_t(i)*pitch;descriptors[7+i*4]=width;
        }
        auto table=tb::tensor(descriptors).to(tb::kCUDA);
        for(int arm:{0,1,0}) {
            const auto option=std::to_string(arm);
#ifdef _WIN32
            _putenv_s("MFQ_MOE_COPY_RANGES",option.c_str());
#else
            setenv("MFQ_MOE_COPY_RANGES",option.c_str(),1);
#endif
            const auto body=[&]{moe_copy_mapped(table.data_ptr(),reinterpret_cast<uint32_t*>(cancel.data_ptr()),blocks,threads,stream);};
            for(bool aborted:{true,false}) {
                destination.fill_(255);cancel.fill_(aborted?1:0);body();
                auto result=destination.cpu();const auto* actual=result.data_ptr<uint8_t>();
                for(int i=0;i<bytes;++i) {
                    const int row=i/pitch,within=i%pitch;
                    const bool copied=!aborted && row<fields && within>=misalign && within<misalign+width;
                    check(actual[i]==(copied?values[row*pitch+within-misalign]:255),"mapped field copy changed data or padding");
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
            std::cout<<"mapped_copy fields="<<fields<<" width="<<width<<" ranges="<<arm
                <<" us="<<ms*1000/(64*16)<<"\n";
        }
    }
    context->end_graph_pool(stream);
    std::cout<<"mapped_copy unaligned destinations, descriptor gaps, tails, cancellation PASS"<<std::endl;
}
