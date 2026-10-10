#pragma once
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::cuda {
struct DmaCopy {
    void* destination=nullptr;
    const void* source=nullptr;
    std::size_t bytes=0;
};

// CUDA 12.x counterpart to the independent pinned transfers batched by
// Strata's core/dma_batch.hpp with CUDA 13's cudaMemcpyBatchAsync. Sources
// and destinations retain their original compressed bytes and addresses.
class DmaCopyBatch {
    struct Entry {
        cudaGraph_t graph=nullptr;
        cudaGraphExec_t execution=nullptr;
        std::vector<cudaGraphNode_t> nodes;
        ~Entry() {
            if(execution)(void)cudaGraphExecDestroy(execution);
            if(graph)(void)cudaGraphDestroy(graph);
        }
    };
    cudaStream_t stream_;
    std::vector<std::unique_ptr<Entry>> cache_;
    bool submitted_=false;
    static void checked(cudaError_t result) {
        if(result!=cudaSuccess)throw std::runtime_error(std::string("CUDA DMA batch: ")+cudaGetErrorString(result));
    }
    static void trace(const char* stage,std::size_t count) {
        const auto* enabled=std::getenv("MFQ_TRACE_MOE_DMA_GRAPH");
        if(enabled && enabled[0]=='1')std::fprintf(stderr,"dma_graph_stage=%s copies=%zu\n",stage,count);
    }
public:
    explicit DmaCopyBatch(cudaStream_t stream):stream_(stream) {}
    ~DmaCopyBatch() {if(submitted_)(void)cudaStreamSynchronize(stream_);}
    DmaCopyBatch(const DmaCopyBatch&)=delete;
    DmaCopyBatch& operator=(const DmaCopyBatch&)=delete;
    // Build before a consumer graph starts. Instantiation while that graph
    // awaits host notification can deadlock with its allocation nodes under
    // WDDM. Replay only updates existing nodes; unknown counts use the loop.
    void prepare(std::size_t maximum_count,void* destination,const void* source) {
        if(!cache_.empty())throw std::logic_error("CUDA DMA batch was already prepared");
        if(maximum_count>128 || !destination || !source)throw std::invalid_argument("invalid CUDA DMA preparation");
        for(std::size_t count=1;count<=maximum_count;++count) {
            auto fresh=std::make_unique<Entry>();trace("create",count);checked(cudaGraphCreate(&fresh->graph,0));
            fresh->nodes.resize(count);trace("add_nodes",count);
            for(std::size_t i=0;i<count;++i)
                checked(cudaGraphAddMemcpyNode1D(&fresh->nodes[i],fresh->graph,nullptr,0,
                    destination,source,1,cudaMemcpyHostToDevice));
            trace("instantiate",count);checked(cudaGraphInstantiate(&fresh->execution,fresh->graph,0));
            checked(cudaGraphUpload(fresh->execution,stream_));
            checked(cudaGraphLaunch(fresh->execution,stream_));submitted_=true;
            cache_.push_back(std::move(fresh));
        }
        checked(cudaStreamSynchronize(stream_));
    }
    bool submit(const std::vector<DmaCopy>& copies) {
        if(copies.empty())return false;
        for(const auto& copy:copies)
            if(!copy.destination || !copy.source || !copy.bytes)
                throw std::invalid_argument("empty field in CUDA DMA batch");
        auto found=std::find_if(cache_.begin(),cache_.end(),[&](const auto& entry) {
            return entry->nodes.size()==copies.size();
        });
        if(found==cache_.end()) {
            trace("ordinary",copies.size());
            submitted_=true;
            for(const auto& copy:copies)checked(cudaMemcpyAsync(copy.destination,copy.source,copy.bytes,cudaMemcpyHostToDevice,stream_));
            return false;
        }
        auto* entry=found->get();trace("update",copies.size());
        for(std::size_t i=0;i<copies.size();++i)
            checked(cudaGraphExecMemcpyNodeSetParams1D(entry->execution,entry->nodes[i],
                copies[i].destination,copies[i].source,copies[i].bytes,cudaMemcpyHostToDevice));
        trace("launch",copies.size());submitted_=true;checked(cudaGraphLaunch(entry->execution,stream_));
        trace("submitted",copies.size());return true;
    }
};
}
