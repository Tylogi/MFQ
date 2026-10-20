#pragma once
#include "kernels/moe/pipeline.h"

static void benchmark_decode_kv_cache() {
    namespace tb=mfq_tensor_backend;
    auto handle=mfq_get_stream_from_pool();MfqCudaStreamGuard guard(handle);
    auto context=default_context();const auto stream=handle.stream();context->begin_graph_pool(stream);
    const auto setting=[](const char* name,bool enabled) {
#ifdef _WIN32
        _putenv_s(name,enabled?"1":"0");
#else
        setenv(name,enabled?"1":"0",1);
#endif
    };
    const auto measure=[&](const auto& body) {
        constexpr int repeat=64,replays=128;
        {GraphWarmupScope warmup(context,stream);body();}
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
        Graph graph;graph.capture_begin_shared();
        for(int i=0;i<repeat;++i)body();
        graph.capture_end();
        for(int i=0;i<8;++i)graph.replay();
        cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
        MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int i=0;i<replays;++i)graph.replay();
        MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
        MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
        return double(ms)*1000/(repeat*replays);
    };
    {
        constexpr int maximum=65536,cells=51551,heads=2,width=128,iw=128;
        const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
        auto hot=HotKvCache::allocate(heads,maximum,cells,width,width,options);
        hot.set_cold_storage(tb::empty({1,heads,maximum-cells,width},options),
            tb::empty({1,heads,maximum-cells,width},options));
        auto stored=tb::empty({1,maximum,iw},options);
        for(int rows:{1,2,4,6,8,16,32})for(bool floating:{false,true}) {
            const auto source_options=options.dtype(floating?tb::kFloat32:tb::kFloat16);
            auto k=tb::ones({1,rows,heads,width+32},source_options).narrow(3,0,width).permute({0,2,1,3});
            auto v=tb::ones({1,rows,heads,width+32},source_options).narrow(3,0,width).permute({0,2,1,3});
            auto raw=tb::ones({1,rows,iw*3},options).narrow(2,iw,iw);
            auto positions=tb::arange(rows,options.dtype(tb::kInt64));
            for(int arm:{0,1,0}) {
                setting("MFQ_KV_APPEND_FUSED",arm!=0);
                const double us=measure([&] {
                    if(!hot.append_decode(k,v,positions,raw,stored)) {
                        hot.append(k.to(tb::kFloat16),v.to(tb::kFloat16),positions);stored.index_copy_(1,positions,raw);
                    }
                });
                std::cout<<"kv_append_bench rows="<<rows<<" float_input="<<floating<<" fused="<<arm<<" us="<<us<<"\n";
            }
        }
        // Resident prefix and long-context hit marking. Same selection is read
        // repeatedly, so this measures metadata work without PCIe variation.
        hot.reset();
        auto positions=tb::arange(4096,options.dtype(tb::kInt64));
        auto full=tb::ones({1,heads,4096,width},options);hot.append(full,full,positions);
        auto source=tb::empty({1,heads,maximum,width},options);
        for(int rows:{1,2,4,6})for(bool short_prefix:{true,false}) {
            auto selected=tb::arange(rows*2047,options.dtype(tb::kInt32)).remainder(2047).reshape({1,rows,2047});
            auto pos=tb::arange(rows,positions.options())+(short_prefix?4000:60000);
            auto key=full.narrow(2,0,rows),raw=tb::ones({1,rows,iw},options);
            for(int arm:{0,1,0}) {
                setting("MFQ_KV_PREPARE_FUSED",arm!=0);
                setting("MFQ_KV_APPEND_FUSED",true);
                const double us=measure([&]{
                    check(hot.append_decode(key,key,pos,raw,stored,arm!=0),"KV benchmark append rejected");
                    hot.prepare(source,source,selected,pos,2047,true,arm!=0);
                });
                std::cout<<"kv_prepare_bench rows="<<rows<<" prefix="<<short_prefix<<" fused="<<arm<<" us="<<us<<"\n";
            }
        }
    }
    setting("MFQ_KV_APPEND_FUSED",true);setting("MFQ_KV_PREPARE_FUSED",true);
    context->end_graph_pool(stream);
}

static void benchmark_route_export() {
    namespace tb=mfq_tensor_backend;
    auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    for(int rows:{1,2,6,32}) {
        const int n=rows*2560,k=rows*10;
        auto host=tb::empty({n+k+1},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32).pinned_memory(true));
        float* alias=nullptr;MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&alias),host.data_ptr(),0));
        auto input=tb::full({n},.25,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
        auto ids=tb::arange(k,input.options().dtype(tb::kInt32));
        for(bool copy:{true,false,true}) {
            auto* host_values=host.data_ptr<float>();std::fill(host_values,host_values+n,-99.f);
            auto* host_ids=reinterpret_cast<int32_t*>(host_values+n);std::fill(host_ids,host_ids+k,-1);
            auto* ready=reinterpret_cast<uint32_t*>(host_values+n+k);*ready=0;
            const auto body=[&] {
                moe_publish_routes(copy?input.data_ptr():nullptr,ids.data_ptr<int32_t>(),copy?alias:nullptr,
                    reinterpret_cast<int32_t*>(alias+n),reinterpret_cast<uint32_t*>(alias+n+k),copy?n:0,k,stream);
            };
            body();MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
            check(*ready==1,"route export did not publish completion");
            for(int i=0;i<k;++i)check(host_ids[i]==i,"route export changed IDs");
            for(int i=0;i<n;++i)check(host_values[i]==(copy?.25f:-99.f),"route export copied an unexpected CPU input");
            Graph graph;graph.capture_begin_shared();for(int i=0;i<64;++i)body();graph.capture_end();
            for(int i=0;i<8;++i)graph.replay();
            cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
            MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));for(int i=0;i<128;++i)graph.replay();
            MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
            std::cout<<"moe_route_export rows="<<rows<<" cpu_input="<<copy<<" us="<<ms*1000/(64*128)<<"\n";
        }
    }
    context->end_graph_pool(stream);
}
