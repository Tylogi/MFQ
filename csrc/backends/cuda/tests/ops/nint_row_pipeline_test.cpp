#include "native/env.h"
#include "storage/weights/nint_row_pipeline.h"
#include "ops/decode_window.h"
#include "ops/quant/nint.h"
#include "kernels/moe/pipeline.h"
#include "mfq/quant/nint_row_cache.h"
#include "../../../../tests/quant/nint_row_fixture.h"
#include <atomic>
#include <cstring>
#include <iostream>
#include <cstdlib>
namespace tb=mfq_tensor_backend;
void early_prefix_rows() {
    auto fixture=std::make_shared<mfq::test::Fixture>(mfq::test::fixture(529,160,28,7,false));
    std::atomic<int> submissions{0};
    std::vector<std::shared_ptr<mfq::NintRows>> tables;
    for(int shard=0;shard<4;++shard)tables.push_back(std::make_shared<mfq::NintRows>(fixture->blob.size(),
        [fixture](std::size_t off,uint8_t* dst,std::size_t n){std::memcpy(dst,fixture->blob.data()+off,n);},true,
        [fixture,&submissions](const std::vector<mfq::ReadSpan>& spans) {
            ++submissions;
            for(const auto& span:spans)std::memcpy(span.destination,fixture->blob.data()+span.offset,span.size);
        }));
    const tb::Device device{tb::kCUDA,0};
    for(bool matches:{true,false}) {
        NintRowPipeline pipeline(tables);
        const std::vector<int64_t> prefix={1,529+2,529*2+3,529*3+4};
        auto ids=prefix;
        ids.insert(ids.end(),{5,529+2,529*2+6,529*3+4});
        if(!matches)ids[0]=7;
        const auto before=submissions.load();
        pipeline.issue(prefix,{1,1,4},true);
        auto actual=pipeline.collect(ids,{1,2,4},device).cpu().contiguous();
        // Four shards for the hint; only shards zero and two have new rows.
        if(submissions.load()-before!=6)
            throw std::runtime_error("PLE prefix was reread or extension missed new rows");
        auto expected=nint_sharded_embedding_lookup(tables,ids,{1,2,4},device).cpu().contiguous();
        if(actual.sizes()!=expected.sizes() ||
           std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
            throw std::runtime_error("PLE early prefix changed extended-window output");
        auto stage=pipeline.make_stage({1,2,4},device);
        pipeline.upload(*stage,ids,{1,2,4});stage->decode();
        auto staged=stage->output().cpu().contiguous();
        if(std::memcmp(staged.data_ptr(),expected.data_ptr(),staged.numel()*staged.element_size()))
            throw std::runtime_error("PLE early prefix changed graph-stage payload");
    }
    {
        NintRowPipeline pipeline(tables);
        const std::vector<int64_t> ids={1,531,1061,1591,5,531,1064,1591,1,537,1061,1599};
        const auto before=submissions.load();
        pipeline.issue(std::vector<int64_t>(ids.begin(),ids.begin()+4),{1,1,4},true);
        pipeline.issue(std::vector<int64_t>(ids.begin(),ids.begin()+8),{1,2,4},true);
        auto actual=pipeline.collect(ids,{1,3,4},device).cpu().contiguous();
        if(submissions.load()-before!=8)
            throw std::runtime_error("PLE successive draft prefixes duplicated a row read");
        auto expected=nint_sharded_embedding_lookup(tables,ids,{1,3,4},device).cpu().contiguous();
        if(actual.sizes()!=expected.sizes() ||
           std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
            throw std::runtime_error("PLE successive draft prefixes changed output");
    }
    // Abandoned hints finish safely without any subsequent target upload.
    {NintRowPipeline pipeline(tables);pipeline.issue({9,530,1061,1591},{1,1,4},true);}
    std::cout<<"MFQ PLE early-prefix extension/chain/mismatch/abandon PASS cases=4\n";
}
void large_window() {
    auto fixture=std::make_shared<mfq::test::Fixture>(mfq::test::fixture(529,160,28,7,false));
    auto table=std::make_shared<mfq::NintRows>(fixture->blob.data(),fixture->blob.size());
    NintRowPipeline pipeline({table});const tb::Device device{tb::kCUDA,0};
    const std::vector<int64_t> shape={1,2,4},ids={0,528,1,257,255,0,257,528};
    auto stage=pipeline.make_stage(shape,device);
    auto expected=pipeline.collect(ids,shape,device).to(tb::kCPU).contiguous();
    constexpr int layers=48;
    mfq::cuda::HostBuffer flags(layers*128,true);std::memset(flags.data(),0,flags.size());
    auto* host=static_cast<uint32_t*>(flags.data());uint32_t* mapped=nullptr;
    MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped),host,0));
    for(int i=0;i<layers;++i)mfq::cuda::publish_mapped_flag(host+i*32+16);
    const auto compute=mfq_current_cuda_stream();auto dma=mfq_get_stream_from_pool();
    mfq::cuda::DecodeWindow window(compute);
    window.capture([&] {
        for(int layer=0;layer<layers;++layer) {
            auto* h=host+layer*32;auto* d=mapped+layer*32;
            window.enroll({h,[&,h]{std::memset(h,0,128);pipeline.issue(ids,shape);},[&,h] {
                mfq::cuda::wait_route_publication(h,compute);
                MfqCudaStreamGuard guard(dma);pipeline.upload(*stage,ids,shape);
                if(stage->mapped_rows())mfq::cuda::publish_mapped_flag(h+16);
                else MFQ_CUDA_CHECK(cudaLaunchHostFunc(dma.stream(),[](void* p){
                    mfq::cuda::publish_mapped_flag(static_cast<uint32_t*>(p));
                },h+16));
            },{},[h]{mfq::cuda::publish_mapped_flag(h+16);}});
            mfq::cuda::signal_mapped_flag(d,compute);mfq::cuda::wait_mapped_flag(d+16,compute);
            stage->decode();
            for(int work=0;work<256;++work)mfq::cuda::signal_mapped_flag(d+1,compute);
        }
    },[]{});
    std::cout<<"MFQ large row window captured nodes="<<window.nodes()<<std::endl;
    for(int i=0;i<2;++i)window.run();
    auto actual=stage->output().to(tb::kCPU).contiguous();
    if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
        throw std::runtime_error("MFQ large row window differs");
    std::cout<<"MFQ large row window PASS layers="<<layers<<std::endl;
}
int main(int argc,char** argv)try {
    const bool mapped_rows=argc==2 && std::string(argv[1])=="--mapped-rows";
    const bool device_rows=argc==2 && std::string(argv[1])=="--device-rows";
    if(mapped_rows || device_rows) {
        mfq::cuda::env_set("MFQ_PLE_MAPPED_ROWS",mapped_rows?"1":"0");
    }
    // Hold the context before borrowing its raw stream. current_stream()'s
    // temporary owner otherwise destroys a freshly created context here;
    // the usual driver often reuses the handle, hiding the stale stream.
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    early_prefix_rows();
    if(!mfq::cuda::env_lookup("MFQ_PLE_MAPPED_ROWS")) {
        auto fixture=std::make_shared<mfq::test::Fixture>(mfq::test::fixture(529,160,28,7,false));
        auto table=std::make_shared<mfq::NintRows>(fixture->blob.data(),fixture->blob.size());
        NintRowPipeline pipeline({table});const tb::Device device{tb::kCUDA,0};
        if(!pipeline.make_stage({1,1,4},device)->mapped_rows() ||
           !pipeline.make_stage({1,6,4},device)->mapped_rows() ||
           !pipeline.make_stage({24},device)->mapped_rows() ||
           pipeline.make_stage({1536},device)->mapped_rows() ||
           pipeline.make_stage({1,32,4},device)->mapped_rows() ||
           pipeline.make_stage({1,6,256},device)->mapped_rows())
            throw std::runtime_error("PLE automatic packet policy changed");
    }
    if(argc==2 && std::string(argv[1])=="--large-window") {
        large_window();context->end_graph_pool(stream);return 0;
    }
    {
        mfq::NintRowCache cache(2);
        auto a=std::make_shared<mfq::NintRowBatch>(), b=std::make_shared<mfq::NintRowBatch>();
        auto c=std::make_shared<mfq::NintRowBatch>(), d=std::make_shared<mfq::NintRowBatch>();
        cache.insert(1,a);cache.insert(2,b);
        auto leased=cache.find(1);
        cache.insert(3,c);
        if(cache.find(2) || cache.find(1)!=a || cache.find(3)!=c)
            throw std::runtime_error("canonical row LRU evicted a recently used entry");
        cache.insert(4,d);
        if(cache.find(1) || leased!=a || cache.find(4)!=d)
            throw std::runtime_error("row cache eviction invalidated a consumer lease");
    }
    int cases=0,graph_cases=0,changed_rows=0,cancel_guards=0,recoveries=0;
    for(const auto sub_bits:{4,5,6,7,8})for(const bool adaptive:{false,true}) {
        // The synthetic adaptive fixture uses every k selector. Canonical
        // streams cannot select k=9 or 10; static GR8/k7 remains covered.
        if(adaptive && sub_bits>6)continue;
        auto fixture=std::make_shared<mfq::test::Fixture>(mfq::test::fixture(529,160,28,sub_bits,adaptive));
        std::atomic<uint64_t> submissions{0};
        std::vector<std::shared_ptr<mfq::NintRows>> tables;
        for(int shard=0;shard<4;++shard)tables.push_back(std::make_shared<mfq::NintRows>(fixture->blob.size(),
            [fixture](std::size_t off,uint8_t* dst,std::size_t n){std::memcpy(dst,fixture->blob.data()+off,n);},true,
            [fixture,&submissions](const std::vector<mfq::ReadSpan>& spans){++submissions;for(const auto& s:spans)std::memcpy(s.destination,fixture->blob.data()+s.offset,s.size);}));
        NintRowPipeline pipeline(tables);
        std::vector<int64_t> ids={0,528,529+1,529*3+257,529*2+255,0,529*3+257,528};
        const std::vector<int64_t> shape={1,2,4};const tb::Device device{tb::kCUDA,0};
        auto stage=pipeline.make_stage(shape,device);
        const auto* output_address=stage->output().data_ptr();
        mfq::cuda::HostBuffer flags(128,true);std::memset(flags.data(),0,128);
        auto* host=static_cast<uint32_t*>(flags.data());uint32_t* mapped=nullptr;
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped),host,0));
        mfq::cuda::publish_mapped_flag(host+16);
        const auto compute=mfq_current_cuda_stream();auto dma=mfq_get_stream_from_pool();
        mfq::cuda::DecodeWindow window(compute);
        const auto capture_rows=[&] {
            window.enroll({stage.get(),[&] {
                std::memset(host,0,128);pipeline.issue(ids,shape);
            },[&] {
                mfq::cuda::wait_route_publication(host,compute);
                MfqCudaStreamGuard guard(dma);pipeline.upload(*stage,ids,shape);
                if(stage->mapped_rows())mfq::cuda::publish_mapped_flag(host+16);
                else MFQ_CUDA_CHECK(cudaLaunchHostFunc(dma.stream(),[](void* p){
                    mfq::cuda::publish_mapped_flag(static_cast<uint32_t*>(p));
                },host+16));
            },{},[&]{mfq::cuda::publish_mapped_flag(host+16);}});
            mfq::cuda::signal_mapped_flag(mapped,compute);
            mfq::cuda::wait_mapped_flag(mapped+16,compute);
            stage->decode();
        };
        window.capture(capture_rows,[]{});
        for(int order=0;order<3;++order) {
            if(order)std::reverse(ids.begin(),ids.end());
            pipeline.issue(ids,shape);
            auto actual=pipeline.collect(ids,shape,device).to(tb::kCPU).contiguous();
            const auto after=submissions.load();
            if((order==0 && after!=4) || (order>0 && after!=4+uint64_t(order)*4))
                throw std::runtime_error("PLE cache did not reuse canonical rows on reordered requests");
            auto expected=nint_sharded_embedding_lookup(tables,ids,shape,device).to(tb::kCPU).contiguous();
            if(actual.sizes()!=expected.sizes() || std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                throw std::runtime_error("MFQ PLE issue/collect sparse shard output differs");
            ++cases;
            window.run();
            auto graph=stage->output().to(tb::kCPU).contiguous();
            if(stage->output().data_ptr()!=output_address || graph.sizes()!=expected.sizes() ||
                std::memcmp(graph.data_ptr(),expected.data_ptr(),graph.numel()*graph.element_size()))
                throw std::runtime_error("MFQ PLE asynchronous graph decode differs or changed output storage");
            ++graph_cases;
        }
        // Change payloads and descriptor cohorts, not just the inverse order.
        // This also checks visibility when a captured kernel reads host pages
        // rewritten between graph replays.
        for(int step=0;step<16;++step) {
            for(std::size_t i=0;i<ids.size();++i)ids[i]=((step+int(i))%4)*529+(step*31+int(i)*17)%529;
            auto expected=nint_sharded_embedding_lookup(tables,ids,shape,device).cpu().contiguous();
            window.run();auto actual=stage->output().cpu().contiguous();
            if(actual.sizes()!=expected.sizes() || std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                throw std::runtime_error("PLE changed-row replay saw stale payload or descriptors");
            ++changed_rows;
        }
        const auto valid_ids=ids;ids[0]=-1;
        bool rejected=false;
        try{window.run();}catch(const std::out_of_range&){rejected=true;++cancel_guards;}
        if(!rejected)throw std::runtime_error("PLE invalid gather was not rejected");
        rejected=false;
        try{window.run();}catch(const std::runtime_error&){rejected=true;++cancel_guards;}
        if(!rejected)throw std::runtime_error("PLE reused a failed decode window");
        ids=valid_ids;
        mfq::cuda::DecodeWindow fresh(compute);
        // Re-enroll using the new window while retaining the same row stage.
        fresh.capture([&] {
            fresh.enroll({stage.get(),[&]{std::memset(host,0,128);pipeline.issue(ids,shape);},[&] {
                mfq::cuda::wait_route_publication(host,compute);
                MfqCudaStreamGuard guard(dma);pipeline.upload(*stage,ids,shape);
                if(stage->mapped_rows())mfq::cuda::publish_mapped_flag(host+16);
                else MFQ_CUDA_CHECK(cudaLaunchHostFunc(dma.stream(),[](void* p){
                    mfq::cuda::publish_mapped_flag(static_cast<uint32_t*>(p));
                },host+16));
            },{},[&]{mfq::cuda::publish_mapped_flag(host+16);}});
            mfq::cuda::signal_mapped_flag(mapped,compute);
            mfq::cuda::wait_mapped_flag(mapped+16,compute);stage->decode();
        },[]{});
        fresh.run();auto recovered=stage->output().cpu().contiguous();
        auto expected=nint_sharded_embedding_lookup(tables,ids,shape,device).cpu().contiguous();
        if(std::memcmp(recovered.data_ptr(),expected.data_ptr(),recovered.numel()*recovered.element_size()))
            throw std::runtime_error("PLE recovery retained incomplete row data");
        ++recoveries;
    }
    // The 12K-node submission stress is checked separately without kernel
    // instrumentation; retain all row correctness/recovery cases in memcheck.
    if(!mfq::cuda::env_lookup("MFQ_PLE_TEST_SKIP_LARGE_WINDOW"))large_window();
    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));context->end_graph_pool(stream);
    std::cout<<"MFQ PLE native row gather/cache/issue/collect PASS cases="<<cases
        <<" asynchronous graph cases="<<graph_cases<<" changed_rows="<<changed_rows
        <<" cancel_guards="<<cancel_guards<<" recoveries="<<recoveries<<" mapped_rows="<<mapped_rows<<std::endl;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
