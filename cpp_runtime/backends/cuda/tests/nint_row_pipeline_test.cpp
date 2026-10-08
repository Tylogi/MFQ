#include "runtime/nint_row_pipeline.h"
#include "runtime/decode_window.h"
#include "nint.h"
#include "runtime/moe_pipeline.h"
#include "mfq/nint_row_cache.h"
#include "../../../tests/nint_row_fixture.h"
#include <atomic>
#include <cstring>
#include <iostream>
namespace tb=mfq_tensor_backend;
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
                MFQ_CUDA_CHECK(cudaLaunchHostFunc(dma.stream(),[](void* p){
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
    const auto stream=mfq_current_cuda_stream();
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());context->begin_graph_pool(stream);
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
    int cases=0,graph_cases=0;
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
        window.capture([&] {
            window.enroll({stage.get(),[&] {
                std::memset(host,0,128);pipeline.issue(ids,shape);
            },[&] {
                mfq::cuda::wait_route_publication(host,compute);
                MfqCudaStreamGuard guard(dma);pipeline.upload(*stage,ids,shape);
                MFQ_CUDA_CHECK(cudaLaunchHostFunc(dma.stream(),[](void* p){
                    mfq::cuda::publish_mapped_flag(static_cast<uint32_t*>(p));
                },host+16));
            },{},[&]{mfq::cuda::publish_mapped_flag(host+16);}});
            mfq::cuda::signal_mapped_flag(mapped,compute);
            mfq::cuda::wait_mapped_flag(mapped+16,compute);
            stage->decode();
        },[]{});
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
    }
    large_window();
    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));context->end_graph_pool(stream);
    std::cout<<"MFQ PLE native row gather/cache/issue/collect PASS cases="<<cases
        <<" asynchronous graph cases="<<graph_cases<<'\n';
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
