#pragma once
#include "mfe_ffn_prepare.h"
#include "mfe_expert_ffn_test.h"
#include "runtime/moe_pipeline.h"

static void benchmark_mfe_prepare(bool timings=true) {
    namespace tb=mfq_tensor_backend;
    auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    mfe_expert_plan_test::verify(timings,stream);
    mfe_expert_plan_test::verify_cache(timings,stream);
    int cases=0;
    for(int tokens:{1,2,6,32})for(int width:{31,640,2559,2560})
    for(int gs:{1,4,7,16,24,28,32,33,48,64}) {
        auto host=tb::empty({tokens,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        for(int i=0;i<tokens*width;++i)host.data_ptr<float>()[i]=i%4?float((i*37)%491-245)/37.0f:0.0f;
        auto input=host.to(tb::kCUDA).to(tb::kFloat16);
        MfeFfnBatch batch;batch.tokens=tokens;batch.routes=10;batch.input_width=width;batch.scale_stride=27;
        batch.input=reinterpret_cast<const __half*>(input.data_ptr());
        auto completion=tb::empty({tokens*batch.routes*batch.scale_stride},input.options().dtype(tb::kInt32));
        batch.completion=reinterpret_cast<uint32_t*>(completion.data_ptr());
        const int groups=(width+gs-1)/gs,other_gs=gs==24?28:24,other_groups=(width+other_gs-1)/other_gs;
        auto codes=tb::empty({tokens*(groups*gs+other_groups*other_gs)},input.options().dtype(tb::kInt8));
        auto scales=tb::empty({tokens*(groups+other_groups)},input.options().dtype(tb::kFloat32));
        MfeInputQuantization descriptors[2]{{codes.data_ptr<int8_t>(),scales.data_ptr<float>(),groups,gs},
            {codes.data_ptr<int8_t>()+tokens*groups*gs,scales.data_ptr<float>()+tokens*groups,other_groups,other_gs}};
        auto descriptor_host=tb::empty({int64_t(sizeof(descriptors))},host.options().dtype(tb::kUInt8));
        std::memcpy(descriptor_host.data_ptr(),descriptors,sizeof(descriptors));
        auto table=descriptor_host.to(tb::kCUDA);
        tb::Tensor reference_codes,reference_scales;
        for(int arm:{0,1,0}) {
            const auto body=[&] {
                const auto* inputs=reinterpret_cast<const MfeInputQuantization*>(table.data_ptr());
                if(arm)mfe_ffn_prepare_warp(batch,inputs,2,groups+other_groups,stream);
                else mfe_ffn_prepare(batch,inputs,2,groups+other_groups,stream);
            };
            completion.fill_(127);codes.fill_(99);scales.fill_(99);body();
            auto actual_codes=codes.cpu(),actual_scales=scales.cpu(),actual_completion=completion.cpu();
            for(int i=0;i<actual_completion.numel();++i)check(actual_completion.data_ptr<int32_t>()[i]==0,"MFE prepare completion not reset");
            if(!reference_codes.defined()){reference_codes=actual_codes;reference_scales=actual_scales;}
            else {
                check(!std::memcmp(reference_codes.data_ptr(),actual_codes.data_ptr(),actual_codes.nbytes()),"MFE warp prepare quantization changed");
                check(!std::memcmp(reference_scales.data_ptr(),actual_scales.data_ptr(),actual_scales.nbytes()),"MFE warp prepare scales changed");
            }
            if(!timings || width!=2560 || gs!=24)continue;
            Graph graph;graph.capture_begin_shared();for(int i=0;i<32;++i)body();graph.capture_end();
            for(int i=0;i<8;++i)graph.replay();
            cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
            MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));for(int i=0;i<64;++i)graph.replay();
            MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
            std::cout<<"mfe_prepare tokens="<<tokens<<" warp="<<arm<<" us="<<ms*1000/(64*32)<<"\n";
        }
        if(width==2560 && gs==24) {
            const int pairs=tokens*batch.routes;
            std::vector<int32_t> routes(pairs);for(int i=0;i<pairs;++i)routes[i]=(i*7)%31;
            auto ids=tb::tensor(routes).to(tb::kCUDA);batch.ids=ids.data_ptr<int32_t>();
            HostBuffer publication((pairs+16)*sizeof(int32_t),true);int32_t* mapped=nullptr;
            MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped),publication.data(),0));
            auto* ready=reinterpret_cast<uint32_t*>(mapped+pairs);
            for(int arm:{0,1,0}) {
                std::memset(publication.data(),0xff,publication.size());
                const auto body=[&]{
                    if(arm)mfe_ffn_prepare_publish(batch,reinterpret_cast<const MfeInputQuantization*>(table.data_ptr()),
                        2,groups+other_groups,mapped,ready,stream);
                    else {
                        moe_publish_routes(nullptr,batch.ids,nullptr,mapped,ready,0,pairs,stream);
                        mfe_ffn_prepare(batch,reinterpret_cast<const MfeInputQuantization*>(table.data_ptr()),2,groups+other_groups,stream);
                    }
                };
                body();MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                check(!std::memcmp(publication.data(),routes.data(),pairs*sizeof(int32_t)),"MFE fused publication changed routes");
                check(static_cast<uint32_t*>(publication.data())[pairs]==1,"MFE fused publication lost ready flag");
                if(!timings)continue;
                Graph graph;graph.capture_begin_shared();for(int i=0;i<16;++i)body();graph.capture_end();
                for(int i=0;i<8;++i)graph.replay();
                cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
                MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));for(int i=0;i<32;++i)graph.replay();
                MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
                float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
                MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
                std::cout<<"mfe_publication_prepare tokens="<<tokens<<" fused="<<arm<<" us="<<ms*1000/(32*16)<<"\n";
            }
        }
        ++cases;
    }
    context->end_graph_pool(stream);
    std::cout<<"mfe_prepare mixed GS1..64, tails, completion byte-exact cases="<<cases<<" PASS"<<std::endl;
}
