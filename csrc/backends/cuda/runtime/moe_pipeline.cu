#include "moe_pipeline.h"
#include <cuda_fp16.h>
#include <cuda/atomic>

namespace mfq::cuda {
namespace {
__device__ void publish_host_flag(std::uint32_t* flag) {
    // Aligned stores also work on mapped host memory without native PCIe
    // atomics. A read-modify-write exchange requires that extra capability.
    ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system> ready(*flag);
    ready.store(1u, ::cuda::std::memory_order_release);
}
__global__ void export_routes(const __half* input, const std::int32_t* routes,
        float* host_input, std::int32_t* host_routes, std::uint32_t* ready,
        int input_count, int route_count) {
    const auto total = static_cast<long long>(input_count) + route_count;
    for (auto item = static_cast<long long>(threadIdx.x); item < total; item += blockDim.x) {
        if (item < input_count) host_input[item] = __half2float(input[item]);
        else host_routes[item - input_count] = routes[item - input_count];
    }
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) publish_host_flag(ready);
}
__global__ void await_host(const std::uint32_t* ready) {
    auto* observed = reinterpret_cast<const volatile std::uint32_t*>(ready);
    while (!*observed) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
    __threadfence_system();
}
__global__ void notify_host(std::uint32_t* ready) {
    __threadfence_system();
    publish_host_flag(ready);
}
__global__ void await_plan(const std::uint32_t* ready,const std::int32_t* host_kind,
        std::int32_t* device_kind,int experts,const std::uint32_t* host_abort,std::uint32_t* device_abort) {
    if(!threadIdx.x) {
        auto* observed=reinterpret_cast<const volatile std::uint32_t*>(ready);
        while(!*observed) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
        __threadfence_system();
        *device_abort=*reinterpret_cast<const volatile std::uint32_t*>(host_abort);
    }
    __syncthreads();
    for(int expert=threadIdx.x;expert<experts;expert+=blockDim.x)
        device_kind[expert]=reinterpret_cast<const volatile std::int32_t*>(host_kind)[expert];
}
__global__ void select_expert_slots(const std::uint64_t* pointers,int maps,int experts,
        const std::int32_t* kinds,const std::uint32_t* cancelled) {
    const auto count=static_cast<long long>(maps)*experts;
    for(auto index=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;index<count;
            index+=static_cast<long long>(blockDim.x)*gridDim.x) {
        const int expert=static_cast<int>(index%experts);
        const auto* p=pointers+(index/experts)*6;
        const auto* resident=reinterpret_cast<const std::int32_t*>(p[0]);
        const auto* owner=reinterpret_cast<const std::int32_t*>(p[1]);
        const auto* staged=reinterpret_cast<const std::int32_t*>(p[2]);
        auto* resident_output=reinterpret_cast<std::int32_t*>(p[3]);
        auto* staged_output=reinterpret_cast<std::int32_t*>(p[4]);
        auto* retained_output=reinterpret_cast<std::int32_t*>(p[5]);
        const auto aborted = *cancelled;
        const auto kind = kinds[expert];
        resident_output[expert] = !aborted && kind == 1 ? resident[expert] : -1;
        staged_output[expert] = !aborted && kind == 2 && owner[expert] >= 0 && resident[expert] < 0 ? staged[expert] : -1;
        retained_output[expert] = !aborted && kind == 2 ? resident[expert] : -1;
    }
}
__global__ void select_nvq_pool(const std::uint64_t* pointers, int pool_count, int experts,
        std::int32_t* pool_output, std::int32_t* slot_output) {
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= experts) return;
    pool_output[expert] = slot_output[expert] = -1;
    for (int pool = 0; pool < pool_count; ++pool) {
        const auto slot = reinterpret_cast<const std::int32_t*>(pointers[pool])[expert];
        if (slot < 0) continue;
        pool_output[expert] = pool;
        slot_output[expert] = slot;
    }
}
__global__ void classify_resident(const std::uint64_t* pointers,int gate_maps,int up_maps,int down_maps,
        int experts,std::int32_t* kinds,std::uint32_t* cancelled) {
    const int expert=blockIdx.x*blockDim.x+threadIdx.x;
    if(!expert)*cancelled=0;
    if(expert>=experts)return;
    const int counts[3]={gate_maps,up_maps,down_maps};
    bool complete=true;
    int offset=0;
    for(int projection=0;projection<3;++projection) {
        bool resident=false;
        for(int map=offset;map<offset+counts[projection];++map) {
            const auto* slots=reinterpret_cast<const std::int32_t*>(pointers[map*6]);
            resident=resident || slots[expert]>=0;
        }
        complete=complete && resident;
        offset+=counts[projection];
    }
    kinds[expert]=complete?1:0;
}
__global__ void scatter_wire(const std::uint8_t* wire,const std::uint32_t* cancelled,int phase) {
    if(*reinterpret_cast<const volatile std::uint32_t*>(cancelled))return;
    const auto* header=reinterpret_cast<const std::uint64_t*>(wire);
    const auto count=header[0];if(!count)return;
    const auto* descriptors=header+4;
    const auto stride=static_cast<std::uint64_t>(blockDim.x)*gridDim.x*16;
    const auto begin=phase==2?header[3]:header[1];
    const auto end=phase==1?header[3]:header[2];
    for(auto offset=begin+(static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x)*16;
            offset<end;offset+=stride) {
        std::uint64_t low=0,high=count;
        while(low<high) {
            const auto middle=(low+high)/2;
            if(descriptors[middle*3+1]<=offset)low=middle+1;else high=middle;
        }
        if(!low)continue;
        const auto* copy=descriptors+(low-1)*3;
        const auto within=offset-copy[1];const auto bytes=copy[2];if(within>=bytes)continue;
        auto* output=reinterpret_cast<std::uint8_t*>(copy[0])+within;
        const auto address=reinterpret_cast<std::uint64_t>(output);
        if(bytes-within>=16 && (reinterpret_cast<std::uint64_t>(wire+offset)&15)==0 && (address&3)==0) {
            const auto value=*reinterpret_cast<const uint4*>(wire+offset);
            if((address&15)==0)*reinterpret_cast<uint4*>(output)=value;
            else if((address&7)==0) {
                auto* words=reinterpret_cast<uint2*>(output);words[0]=make_uint2(value.x,value.y);words[1]=make_uint2(value.z,value.w);
            } else {
                auto* words=reinterpret_cast<uint32_t*>(output);words[0]=value.x;words[1]=value.y;words[2]=value.z;words[3]=value.w;
            }
        } else for(std::uint64_t byte=0;byte<16 && within+byte<bytes;++byte)output[byte]=wire[offset+byte];
    }
}
__global__ void snapshot_mapped(const uint64_t* source,const uint32_t* cancelled,
        uint64_t* destination,uint64_t capacity) {
    __shared__ uint64_t count;
    __shared__ uint32_t abort;
    if(!threadIdx.x) {
        count=source[0];
        abort=*reinterpret_cast<const volatile uint32_t*>(cancelled) || count>capacity;
        destination[0]=abort?0:count;
        destination[1]=source[1];destination[2]=abort;destination[3]=0;
    }
    __syncthreads();
    if(abort)return;
    for(uint64_t word=threadIdx.x;word<count*4;word+=blockDim.x)
        destination[4+word]=source[4+word];
}
__global__ void copy_mapped(const uint64_t* header,const uint32_t* cancelled) {
    // Host control reads cost a PCIe transaction. Share one snapshot per block
    // instead of issuing a volatile mapped read from every copy thread.
    __shared__ uint32_t abort;
    __shared__ uint64_t count,total;
    if(!threadIdx.x) {
        abort=*reinterpret_cast<const volatile uint32_t*>(cancelled);
        count=header[0];total=header[1];
    }
    __syncthreads();
    if(abort || !count)return;
    const auto* fields=reinterpret_cast<const MoeMappedCopyDescriptor*>(header+4);
    const auto stride=uint64_t(blockDim.x)*gridDim.x*16;
    for(auto offset=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)*16;offset<total;offset+=stride) {
        uint64_t low=0,high=count;
        while(low<high) {
            const auto middle=(low+high)/2;
            if(fields[middle].offset<=offset)low=middle+1;else high=middle;
        }
        if(!low)continue;
        const auto& field=fields[low-1];const auto within=offset-field.offset;
        if(within>=field.bytes)continue;
        auto* output=reinterpret_cast<uint8_t*>(field.destination)+within;
        const auto* input=reinterpret_cast<const uint8_t*>(field.source)+within;
        const auto address=reinterpret_cast<uint64_t>(output);
        if(field.bytes-within>=16 && (reinterpret_cast<uint64_t>(input)&15)==0 && (address&3)==0) {
            const auto value=*reinterpret_cast<const uint4*>(input);
            if((address&15)==0)*reinterpret_cast<uint4*>(output)=value;
            else if((address&7)==0) {
                auto* words=reinterpret_cast<uint2*>(output);
                words[0]=make_uint2(value.x,value.y);words[1]=make_uint2(value.z,value.w);
            } else {
                auto* words=reinterpret_cast<uint32_t*>(output);
                words[0]=value.x;words[1]=value.y;words[2]=value.z;words[3]=value.w;
            }
        } else for(uint64_t byte=0;byte<16 && within+byte<field.bytes;++byte)output[byte]=input[byte];
    }
}
__global__ void plan_nvq_routes(const std::uint64_t* pointers,int pools,int experts,
        const std::int32_t* ids,int pairs,std::int32_t* plan) {
    const int pair=threadIdx.x,lane=pair&31,warp=pair>>5;
    int pool=-1,local=-1;
    if(pair<pairs) {
        const int expert=ids[pair];
        if(static_cast<unsigned>(expert)<static_cast<unsigned>(experts))
            for(int p=0;p<pools;++p) {
                const int slot=reinterpret_cast<const std::int32_t*>(pointers[p])[expert];
                if(slot>=0){pool=p;local=slot;}
            }
        plan[1+pair]=pool;
        plan[1+pairs+pair]=local;
    }
    const unsigned mask=__ballot_sync(0xffffffffu,pair<pairs && pool>=0);
    __shared__ int warp_counts[32];
    if(!lane)warp_counts[warp]=__popc(mask);
    __syncthreads();
    int prefix=0;
    for(int w=0;w<warp;++w)prefix+=warp_counts[w];
    if(pair<pairs && pool>=0)
        plan[1+2*pairs+prefix+__popc(mask&((1u<<lane)-1))]=pair;
    if(!pair) {
        int count=0;
        for(int w=0;w<int(blockDim.x)/32;++w)count+=warp_counts[w];
        plan[0]=count;
    }
}
__global__ void scatter_expert_results(const __half* resident, const __half* staged,
        const std::uint16_t* host, const std::int32_t* routes, const std::int32_t* kinds,
        __half* output, int entries, int width) {
    const auto count = static_cast<long long>(entries) * width;
    const auto stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (auto index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
            index < count; index += stride) {
        const auto kind = kinds[routes[index / width]];
        if (kind == 1) output[index] = resident[index];
        else if (kind == 2) output[index] = staged[index];
        else output[index] = __ushort_as_half(reinterpret_cast<const volatile std::uint16_t*>(host)[index]);
    }
}
}
void moe_publish_routes(const void* input, const std::int32_t* ids, float* host_input,
        std::int32_t* host_ids, std::uint32_t* ready, int n, int k, cudaStream_t stream) {
    export_routes<<<1, 256, 0, stream>>>(static_cast<const __half*>(input), ids,
        host_input, host_ids, ready, n, k);
}
void wait_mapped_flag(const std::uint32_t* flag, cudaStream_t stream) {
    await_host<<<1, 1, 0, stream>>>(flag);
}
void wait_mapped_plan(const std::uint32_t* flag,const std::int32_t* host_kind,std::int32_t* device_kind,
        int experts,const std::uint32_t* host_abort,std::uint32_t* device_abort,cudaStream_t stream) {
    await_plan<<<1,experts ? 256 : 1,0,stream>>>(flag,host_kind,device_kind,experts,host_abort,device_abort);
}
void signal_mapped_flag(std::uint32_t* flag, cudaStream_t stream) {
    notify_host<<<1, 1, 0, stream>>>(flag);
}
void moe_update_dispatch_maps(const std::uint64_t* pointers,int maps,int experts,
        const std::int32_t* kind,const std::uint32_t* abort,cudaStream_t stream) {
    const auto count=static_cast<long long>(maps)*experts;
    if(count)select_expert_slots<<<static_cast<unsigned>((count+255)/256),256,0,stream>>>(
        pointers,maps,experts,kind,abort);
}
void moe_update_nvq_maps(const std::uint64_t* maps, int pools, int experts,
        std::int32_t* pool, std::int32_t* local, cudaStream_t stream) {
    if (experts) select_nvq_pool<<<(experts + 255) / 256, 256, 0, stream>>>(maps, pools, experts, pool, local);
}
void moe_classify_resident(const std::uint64_t* pointers,int gate_maps,int up_maps,int down_maps,
        int experts,std::int32_t* kind,std::uint32_t* abort,cudaStream_t stream) {
    if(experts)classify_resident<<<(experts+255)/256,256,0,stream>>>(
        pointers,gate_maps,up_maps,down_maps,experts,kind,abort);
}
void moe_plan_nvq_routes(const std::uint64_t* maps,int pools,int experts,const std::int32_t* ids,
        int pairs,std::int32_t* plan,cudaStream_t stream) {
    plan_nvq_routes<<<1,unsigned((pairs+31)/32*32),0,stream>>>(maps,pools,experts,ids,pairs,plan);
}
cudaError_t moe_wire_launch_geometry(int* blocks,int* threads) {
    return cudaOccupancyMaxPotentialBlockSize(blocks,threads,scatter_wire,0,0);
}
cudaError_t moe_mapped_copy_launch_geometry(int* blocks,int* threads) {
    return cudaOccupancyMaxPotentialBlockSize(blocks,threads,copy_mapped,0,0);
}
void moe_snapshot_mapped(const void* table,const uint32_t* cancelled,void* snapshot,
        uint64_t capacity,cudaStream_t stream) {
    snapshot_mapped<<<1,256,0,stream>>>(static_cast<const uint64_t*>(table),cancelled,
        static_cast<uint64_t*>(snapshot),capacity);
}
void moe_copy_mapped(const void* table,const uint32_t* cancelled,int blocks,int threads,cudaStream_t stream) {
    copy_mapped<<<blocks,threads,0,stream>>>(static_cast<const uint64_t*>(table),cancelled);
}
void moe_scatter_wire_phase(const void* wire,const std::uint32_t* cancelled,int blocks,int threads,int phase,cudaStream_t stream) {
    scatter_wire<<<blocks,threads,0,stream>>>(static_cast<const std::uint8_t*>(wire),cancelled,phase);
}
void moe_scatter_wire(const void* wire,const std::uint32_t* cancelled,int blocks,int threads,cudaStream_t stream) {
    scatter_wire<<<blocks,threads,0,stream>>>(static_cast<const std::uint8_t*>(wire),cancelled,0);
}
void moe_merge_results(const void* hot, const void* transfer, const void* cpu,
        const std::int32_t* ids, const std::int32_t* kind, void* out,
        int entries, int width, cudaStream_t stream) {
    const auto blocks = static_cast<unsigned>((static_cast<long long>(entries) * width + 255) / 256);
    if (blocks) scatter_expert_results<<<blocks, 256, 0, stream>>>(
        static_cast<const __half*>(hot), static_cast<const __half*>(transfer),
        static_cast<const std::uint16_t*>(cpu), ids, kind, static_cast<__half*>(out), entries, width);
}
}
