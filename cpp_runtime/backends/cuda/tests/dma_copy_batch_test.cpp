#include "runtime/dma_copy_batch.h"
#include "runtime/moe_pipeline.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {
void check(cudaError_t error) {
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
void verify_notification(bool cached,bool registered_source=false) {
    unsigned char* host=nullptr;unsigned char* device=nullptr;
    uint32_t* flags=nullptr;uint32_t* alias=nullptr;
    check(cudaMallocHost(reinterpret_cast<void**>(&host),4096));
    check(cudaMalloc(reinterpret_cast<void**>(&device),4096));
    check(cudaHostAlloc(reinterpret_cast<void**>(&flags),128,cudaHostAllocMapped|cudaHostAllocPortable));
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&alias),flags,0));
    std::memset(host,0x71,4096);std::memset(flags,0,128);
    std::vector<unsigned char> registered(4096,0x23);
    if(registered_source)check(cudaHostRegister(registered.data(),registered.size(),cudaHostRegisterPortable));
    cudaStream_t copy=nullptr,compute=nullptr;
    check(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking));
    cudaGraph_t graph=nullptr;cudaGraphExec_t execution=nullptr;
    check(cudaStreamBeginCapture(compute,cudaStreamCaptureModeThreadLocal));
    void* allocation=nullptr;check(cudaMallocAsync(&allocation,65536,compute));
    check(cudaMemsetAsync(allocation,0xa7,65536,compute));
    mfq::cuda::wait_mapped_flag(alias,compute);
    check(cudaFreeAsync(allocation,compute));
    check(cudaStreamEndCapture(compute,&graph));check(cudaGraphInstantiate(&execution,graph,0));
    bool rescued=false;
    {
        mfq::cuda::DmaCopyBatch batch(copy);
        check(cudaMemcpyAsync(device,host,4096,cudaMemcpyHostToDevice,copy));
        check(cudaStreamSynchronize(copy));
        if(cached)batch.prepare(2,device,host);
        check(cudaGraphLaunch(execution,compute));
        std::atomic<bool> done{false};
        std::jthread watchdog([&] {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while(!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if(!done.load(std::memory_order_acquire)) {
                rescued=true;std::atomic_thread_fence(std::memory_order_seq_cst);
                *static_cast<volatile uint32_t*>(flags)=1;std::atomic_thread_fence(std::memory_order_seq_cst);
            }
        });
        const auto* input=registered_source?registered.data():host;
        const bool graph_used=batch.submit({{device+3,input+7,509},{device+1041,input+1031,263}});
        check(cudaLaunchHostFunc(copy,[](void* value) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *static_cast<volatile uint32_t*>(value)=1;
            std::atomic_thread_fence(std::memory_order_seq_cst);
        },flags));
        const auto progress=cudaStreamQuery(copy);
        if(progress!=cudaSuccess && progress!=cudaErrorNotReady)check(progress);
        check(cudaStreamSynchronize(compute));done.store(true,std::memory_order_release);watchdog.join();
        check(cudaStreamSynchronize(copy));
        if(graph_used!=cached)throw std::runtime_error("notification probe used the wrong submission path");
        std::vector<unsigned char> actual(509);check(cudaMemcpy(actual.data(),device+3,actual.size(),cudaMemcpyDeviceToHost));
        if(std::memcmp(actual.data(),input+7,actual.size()))throw std::runtime_error("notification probe changed copied bytes");
    }
    check(cudaGraphExecDestroy(execution));check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(compute));check(cudaStreamDestroy(copy));
    if(registered_source)check(cudaHostUnregister(registered.data()));
    check(cudaFree(device));check(cudaFreeHost(host));check(cudaFreeHost(flags));
    if(rescued)throw std::runtime_error(std::string("DMA graph blocked an allocation graph waiting for host notification: ")+(cached?"cached update":"unprepared ordinary copy"));
    std::cout<<"DMA graph notification "<<(cached?"cached update":"unprepared ordinary copy")
        <<(registered_source?" registered source":"")<<" exact PASS\n";
}
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--notification-cached") {verify_notification(true);return 0;}
        if(argc==2 && std::string(argv[1])=="--notification-create") {verify_notification(false);return 0;}
        if(argc==2 && std::string(argv[1])=="--notification-registered") {verify_notification(true,true);return 0;}
        constexpr std::size_t region=65536,total=2*region;
        unsigned char* host=nullptr;unsigned char* device=nullptr;
        check(cudaMallocHost(reinterpret_cast<void**>(&host),total));
        check(cudaMalloc(reinterpret_cast<void**>(&device),total));
        cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        for(std::size_t i=0;i<total;++i)host[i]=static_cast<unsigned char>(i*29+(i>>5)*71);
        int cases=0;
        {
            mfq::cuda::DmaCopyBatch batch(stream);
            batch.prepare(16,device,host);
            for(int round=0;round<48;++round) {
                check(cudaMemsetAsync(device,0xa7,total,stream));
                std::vector<unsigned char> expected(total,0xa7),actual(total);
                std::vector<mfq::cuda::DmaCopy> copies;
                const int count=round%24;
                for(int i=0;i<count;++i) {
                    const std::size_t source=(i*1777+round*13)%region;
                    const std::size_t destination=i*2048+(round&15);
                    const std::size_t bytes=1+(i*71+round*19)%1023;
                    copies.push_back({device+destination,host+source,bytes});
                    std::memcpy(expected.data()+destination,host+source,bytes);
                }
                batch.submit(copies);
                check(cudaMemcpyAsync(actual.data(),device,total,cudaMemcpyDeviceToHost,stream));
                check(cudaStreamSynchronize(stream));
                if(actual!=expected)throw std::runtime_error("batched DMA changed payload or neighboring guards");
                ++cases;
            }
            // Updating an executable after launch must preserve the queued
            // invocation's old pointers and sizes as well as the new one's.
            check(cudaMemsetAsync(device,0xa7,total,stream));
            std::vector<unsigned char> expected(total,0xa7),actual(total);
            for(int invocation=0;invocation<2;++invocation) {
                std::vector<mfq::cuda::DmaCopy> copies;
                for(int i=0;i<3;++i) {
                    const auto destination=invocation*region+i*2048+17;
                    const auto source=invocation*region+i*1024+31;
                    const std::size_t bytes=513+invocation*11+i;
                    copies.push_back({device+destination,host+source,bytes});
                    std::memcpy(expected.data()+destination,host+source,bytes);
                }
                batch.submit(copies);
            }
            check(cudaMemcpyAsync(actual.data(),device,total,cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            if(actual!=expected)throw std::runtime_error("DMA executable update changed a queued invocation");
            ++cases;
            bool rejected=false;
            try {batch.submit({{device,host,0}});}catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)throw std::runtime_error("empty DMA field was accepted");
            batch.submit({{device,host,257}});
        }
        std::vector<unsigned char> drained(257);
        check(cudaMemcpy(drained.data(),device,drained.size(),cudaMemcpyDeviceToHost));
        if(std::memcmp(drained.data(),host,drained.size()))throw std::runtime_error("DMA batch destruction did not drain copies");
        check(cudaStreamDestroy(stream));check(cudaFree(device));check(cudaFreeHost(host));
        std::cout<<"dma_copy_batch_cases="<<cases<<" changing count/pointers/sizes, byte guards, ordinary fallback, queued updates and teardown exact PASS\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
