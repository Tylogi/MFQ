#pragma once
#include "native/env.h"
#include "kernels/cache/dma_copy_batch.h"
#include <cuda.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>

namespace mfq::cuda {
// CUDA 13 Driver API, resolved by ABI version while retaining CUDA 12 builds.
// https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__MEM.html
// The Windows source-alias requirement follows Strata's MIT dma_batch.hpp.
class NativeDmaBatch {
    struct Attributes {
        int src_access_order=1; // CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
        CUmemLocation src_location{},dst_location{};
        unsigned flags=0;
    };
    static_assert(sizeof(Attributes)==24 && offsetof(Attributes,flags)==20);
    using Function=CUresult(CUDAAPI*)(CUdeviceptr*,CUdeviceptr*,std::size_t*,std::size_t,
        Attributes*,std::size_t*,std::size_t,CUstream);
    static Function entry() {
#if CUDA_VERSION >= 12000 && CUDART_VERSION >= 12000
        static const auto function=[]() -> Function {
            void* raw=nullptr;
            cudaDriverEntryPointQueryResult query=cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
            const auto result=cudaGetDriverEntryPointByVersion("cuGetProcAddress",&raw,12000,cudaEnableDefault,&query);
#else   // the versioned query arrived in CUDA 12.5
            const auto result=cudaGetDriverEntryPoint("cuGetProcAddress",&raw,cudaEnableDefault,&query);
#endif
            if(result==cudaErrorNotSupported){cudaGetLastError();return nullptr;}
            if(result!=cudaSuccess)throw std::runtime_error("CUDA driver entry query failed: "+std::to_string(result));
            if(query!=cudaDriverEntryPointSuccess || !raw)return nullptr;
            const auto get=reinterpret_cast<decltype(&cuGetProcAddress)>(raw);
            void* batch=nullptr;CUdriverProcAddressQueryResult status=CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
            if(get("cuMemcpyBatchAsync",&batch,13000,CU_GET_PROC_ADDRESS_DEFAULT,&status)!=CUDA_SUCCESS ||
                    status!=CU_GET_PROC_ADDRESS_SUCCESS)return nullptr;
            return reinterpret_cast<Function>(batch);
        }();
        return function;
#else
        return nullptr;
#endif
    }
public:
    static bool available(){return entry()!=nullptr;}
    // Every source has a live pinned owner; callers translate its registered
    // base to the device-visible alias once, before the consumer graph starts.
    template<class Alias>
    static bool submit(const std::vector<DmaCopy>& copies,cudaStream_t stream,int mode,Alias&& alias) {
        if(!mode || copies.empty() || !entry())return false;
        constexpr std::size_t capacity=128;
        if(copies.size()>capacity) {
            const auto* multiple=mfq::cuda::env_lookup("MFQ_MOE_NATIVE_DMA_MULTI_BATCH");
            if(multiple && multiple[0]=='0')return false;
        }
        struct Packet {
            std::array<CUdeviceptr,capacity> destinations{},sources{};
            std::array<std::size_t,capacity> sizes{};
        } first_packet;
        std::vector<Packet> extra_packets((copies.size()-1)/capacity);
        const auto packet=[&](std::size_t index) -> Packet& {
            return index?extra_packets[index-1]:first_packet;
        };
        // Validate every owner and alias before queuing any copy. An invalid
        // field in a later packet must leave the entire fallback untouched.
        for(std::size_t i=0;i<copies.size();++i) {
            const auto& copy=copies[i];
            if(!copy.destination || !copy.source || !copy.bytes)
                throw std::invalid_argument("empty native DMA field");
            auto& p=packet(i/capacity);const auto index=i%capacity;
            p.sources[index]=alias(copy.source,copy.bytes);
            if(!p.sources[index])return false;
            p.destinations[index]=reinterpret_cast<CUdeviceptr>(copy.destination);p.sizes[index]=copy.bytes;
        }
        Attributes attributes;attributes.flags=mode>=2?1u:0u;
        for(std::size_t begin=0;begin<copies.size();begin+=capacity) {
            auto& p=packet(begin/capacity);const auto count=std::min(capacity,copies.size()-begin);
            std::size_t first=0;
            const auto result=entry()(p.destinations.data(),p.sources.data(),p.sizes.data(),count,
                &attributes,&first,1,reinterpret_cast<CUstream>(stream));
            // Once a packet has been queued, a later failure must enter the
            // caller's recovery path, never silently retry a partially queued list.
            if(result==CUDA_ERROR_NOT_SUPPORTED && !begin)return false;
            if(result!=CUDA_SUCCESS) {
                std::string detail="CUDA native DMA batch failed: "+std::to_string(result)+
                    " copies="+std::to_string(copies.size())+" offset="+std::to_string(begin);
                for(std::size_t i=0;i<std::min<std::size_t>(count,12);++i) {
                    const auto& copy=copies[begin+i];
                    cudaPointerAttributes src{},dst{};
                    const auto source_status=cudaPointerGetAttributes(&src,copy.source);
                    const auto destination_status=cudaPointerGetAttributes(&dst,copy.destination);
                    detail+=" ["+std::to_string(begin+i)+" bytes="+std::to_string(p.sizes[i])+" src="+std::to_string(p.sources[i])+
                        " dst="+std::to_string(p.destinations[i])+" type="+std::to_string(int(src.type))+"/"+
                        std::to_string(int(dst.type))+" query="+std::to_string(source_status)+"/"+std::to_string(destination_status)+"]";
                }
                throw std::runtime_error(detail);
            }
        }
        if(!extra_packets.empty()) {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "moe_native_dma_multibatch_dispatch copies=%zu packets=%zu capacity=%zu\n",
                copies.size(),extra_packets.size()+1,capacity);
        }
        return true;
    }
};
}
