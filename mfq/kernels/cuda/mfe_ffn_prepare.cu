#include "mfe_ffn_prepare.h"
#include <cuda/atomic>
#include <cub/block/block_radix_sort.cuh>
#include <cstdlib>
#include <atomic>
#include <cstdio>

namespace mfq::cuda {
namespace {
template<int Items>
__device__ void build_expert_plan(const MfeFfnBatch& b,const MfeExpertPlan& plan) {
    using Sort=cub::BlockRadixSort<uint32_t,64,Items,int32_t>;
    __shared__ typename Sort::TempStorage scratch;
    __shared__ uint32_t sorted_experts[64*Items];
    const int thread=threadIdx.x,entries=b.tokens*b.routes;
    uint32_t experts[Items];
    int32_t pairs[Items];
    for(int expert=thread;expert<b.experts;expert+=64) {
        plan.expert_offsets[expert]=-1;
        plan.expert_counts[expert]=0;
    }
    if(!thread)*plan.unique_count=0;
#pragma unroll
    for(int item=0;item<Items;++item) {
        const int pair=thread*Items+item;
        const int expert=pair<entries?b.ids[pair]:-1;
        experts[item]=unsigned(expert)<unsigned(b.experts)?uint32_t(expert):~uint32_t(0);
        pairs[item]=pair;
    }
    // CUB's stable radix sort preserves original pair order for equal experts.
    // Only expert bits participate; invalid IDs sort after every valid expert.
    Sort(scratch).Sort(experts,pairs,0,32-__clz(unsigned(b.experts)));
#pragma unroll
    for(int item=0;item<Items;++item) {
        const int index=thread*Items+item;
        sorted_experts[index]=experts[item];
        if(index<entries)plan.sorted_pairs[index]=experts[item]<unsigned(b.experts)?pairs[item]:-1;
    }
    __syncthreads();
#pragma unroll
    for(int item=0;item<Items;++item) {
        const int index=thread*Items+item;
        const unsigned expert=experts[item];
        if(expert>=unsigned(b.experts))continue;
        if(!index || sorted_experts[index-1]!=expert) {
            plan.expert_offsets[expert]=index;
            plan.unique_pairs[atomicAdd(plan.unique_count,1)]=pairs[item];
        }
    }
    __syncthreads();
#pragma unroll
    for(int item=0;item<Items;++item) {
        const int index=thread*Items+item;
        const unsigned expert=experts[item];
        if(expert<unsigned(b.experts)) {
            plan.pair_ranks[pairs[item]]=index-plan.expert_offsets[expert];
            if(index+1==entries || sorted_experts[index+1]!=expert)
                plan.expert_counts[expert]=index+1-plan.expert_offsets[expert];
        }else if(pairs[item]<entries)plan.pair_ranks[pairs[item]]=-1;
    }
    __syncthreads();
}
template<bool Ordered>
__device__ void prepare_shared_gate(const MfePrepareSharedGate& gate,int token,int group) {
    const int lane=threadIdx.x&31;
    const auto* x=gate.activation+(int64_t(token)*gate.groups+group)*48;
    const auto* weights=gate.packed+(uint64_t(gate.offset[0])>>3)+group*48;
    int dot=0,sum=0;
    if(lane<12) {
        const int q=reinterpret_cast<const int*>(x)[lane];
        const int code=reinterpret_cast<const int*>(weights)[lane];
        sum=__dp4a(0x01010101,q,0);
        dot=__dp4a(code^int(0x80808080u),q,0)+128*sum;
    }
    const int64_t row=int64_t(token)*gate.groups+group;
    if constexpr(Ordered) {
        if(lane<12)gate.terms[row*12+lane]=gate.outer_scale[0]*float(gate.scales[group])*float(dot)-
            gate.outer_minimum[0]*float(gate.minima[group])*float(sum);
        if(lane<12)__threadfence();
    }else {
#pragma unroll
        for(int step=16;step;step>>=1) {
            dot+=__shfl_xor_sync(0xffffffffu,dot,step);
            sum+=__shfl_xor_sync(0xffffffffu,sum,step);
        }
        if(!lane) {
            gate.terms[row]=gate.outer_scale[0]*float(gate.scales[group])*float(dot)-
                gate.outer_minimum[0]*float(gate.minima[group])*float(sum);
            __threadfence();
        }
    }
    __syncwarp();
    int last=0;
    if(!lane)last=atomicInc(gate.completed+token,unsigned(gate.groups-1))==unsigned(gate.groups-1);
    last=__shfl_sync(0xffffffffu,last,0);
    if(!last)return;
    const auto* xs=reinterpret_cast<const volatile float*>(gate.activation_scales)+int64_t(token)*gate.groups;
    const auto* terms=reinterpret_cast<const volatile float*>(gate.terms)+int64_t(token)*gate.groups*(Ordered?12:1);
    float result=0.f;
    if constexpr(Ordered) {
        const int relative=lane/12,chunk=lane%12;
        if(relative<2)for(int g=relative;g<gate.groups;g+=2)result+=xs[g]*terms[g*12+chunk];
    }else {
        // Reproduce the original four-warp accumulation and reduction order.
#pragma unroll
        for(int warp=0;warp<4;++warp) {
            float partial=0.f;
            if((lane&3)==0)for(int g=warp*8+(lane>>2);g<gate.groups;g+=32)partial+=xs[g]*terms[g];
#pragma unroll
            for(int step=16;step;step>>=1)partial+=__shfl_xor_sync(0xffffffffu,partial,step);
            if(lane==warp)result=partial;
        }
    }
#pragma unroll
    for(int step=16;step;step>>=1)result+=__shfl_xor_sync(0xffffffffu,result,step);
    if(!lane)gate.output[token]=__float2half_rn(1.f/(1.f+expf(-__half2float(__float2half_rn(result)))));
}
__device__ int residency_kind(const MfeFfnBatch& b,int expert) {
    for(int which=0;which<3;++which) {
        // An ungated bank has no Gate cohort; Up and Down own the residency.
        if(which==0 && b.expert_activation==1)continue;
        const int pool=b.cohort_by_expert[which*b.experts+expert];
        if(unsigned(pool)>=unsigned(b.projection_counts[which]))return which==2?6:2;
        const auto& d=b.projections[b.projection_offsets[which]+pool];
        const int local=d.expert_local?d.expert_local[expert]:0;
        if(local<0 || local>=d.local_experts)return which==2?6:2;
    }
    return 1;
}
template<bool Snapshot,bool ResidentPacket=false,bool Descriptors=false>
__global__ void transfer_packet(MfeFfnBatch b,const int32_t* packet,
        const uint64_t* table,uint64_t* snapshot,uint64_t capacity,
        const uint64_t* host_descriptors=nullptr,uint64_t* device_descriptors=nullptr,uint64_t descriptor_capacity=0,
        int32_t* transfer_routes=nullptr,bool early_gu=false) {
    __shared__ uint32_t cancelled;
    __shared__ uint64_t fields;
    __shared__ uint64_t descriptor_words;
    bool resident=false;
    if constexpr(ResidentPacket) {
        // The CPU cannot reassign resident weights while this window runs.
        bool hot=true;
        for(int pair=threadIdx.x;pair<b.tokens*b.routes;pair+=blockDim.x) {
            const int expert=b.ids[pair];
            hot&=unsigned(expert)<unsigned(b.experts) && b.kinds[expert]==1;
        }
        if constexpr(Descriptors)resident=__syncthreads_and(hot);
        else resident=__all_sync(0xffffffffu,hot);
    }
    if(!threadIdx.x) {
        const auto* host_abort=reinterpret_cast<const volatile uint32_t*>(b.host_aborted);
        // The pipeline publishes this packet and plan before submitting any
        // transfer. Its transfer-ready publication covers both; rereading the
        // separate mapped plan flag adds a second PCIe dependency.
        while(!*reinterpret_cast<const volatile uint32_t*>(b.transfer_ready) &&
                !(host_abort && *host_abort))__nanosleep(64);
        cancelled=host_abort && *host_abort?1:0;
        __threadfence_system();
        if constexpr(Snapshot) {
            fields=cancelled || resident?0:table[0];
            const bool empty_packet=resident || fields==~uint64_t(0);
            if(empty_packet)fields=0;
            cancelled|=fields>capacity;
            if constexpr(Descriptors) {
                descriptor_words=cancelled || resident?0:table[2];
                cancelled|=descriptor_words>descriptor_capacity;
            }
            snapshot[0]=cancelled?0:fields;snapshot[1]=cancelled || resident?0:table[1];
            snapshot[2]=cancelled?1u:empty_packet?2u:0u;snapshot[3]=!cancelled && empty_packet;
        }
        if(b.aborted)*b.aborted=cancelled;
    }
    __syncthreads();
    bool cold_route=false;
    for(int pair=threadIdx.x;pair<b.tokens*b.routes;pair+=blockDim.x) {
        const int expert=b.ids[pair];if(unsigned(expert)>=unsigned(b.experts))continue;
        const int previous=b.kinds[expert];
        // Host packets are aligned int2 rows in route order. One coalesced
        // 64-bit read replaces two scattered expert-indexed PCIe reads.
        const auto plan=cancelled?~uint64_t(0):resident?uint64_t(0xffffffff00000001ull):
            *reinterpret_cast<const volatile uint64_t*>(packet+pair*2);
        const int kind=int32_t(uint32_t(plan));
        const int tagged=kind==2 && previous==6?6:kind;
        atomicExch(b.kinds+expert,tagged);
        atomicExch(b.transfer_index+expert,int32_t(uint32_t(plan>>32)));
        cold_route=!cancelled && (tagged&3)==2 && !(early_gu && (tagged&4));
    }
    if constexpr(ResidentPacket && Descriptors)if(transfer_routes && threadIdx.x<32) {
        // M=1, routes<=32: preserve every original slot, including duplicates.
        // This wait already reads the plan, so compaction adds no launch or
        // host readback. The next kernel consumes the completed list.
        const unsigned active=__ballot_sync(0xffffffffu,cold_route);
        const int rank=__popc(active&((1u<<threadIdx.x)-1));
        if(cold_route)transfer_routes[1+rank]=int(threadIdx.x);
        if(!threadIdx.x) {
            transfer_routes[0]=__popc(active);
            // This existing ballot proves every original route was already
            // resident before the transfer wait. Partial GU/Down and CPU
            // routes never qualify, and cancellation retains the old path.
            transfer_routes[b.routes+1]=!cancelled && resident;
        }
    }
    if constexpr(Snapshot)if(!cancelled)
        for(uint64_t word=threadIdx.x;word<fields*4;word+=blockDim.x)snapshot[4+word]=table[4+word];
    // The same release/acquire publication protects route choices, loan
    // descriptors and these immutable weight views. Copy the views once into
    // VRAM; arithmetic CTAs never reread the mapped host descriptor image.
    if constexpr(Descriptors)if(!cancelled) {
        for(uint64_t vector=threadIdx.x;vector<descriptor_words/2;vector+=blockDim.x)
            reinterpret_cast<uint4*>(device_descriptors)[vector]=reinterpret_cast<const uint4*>(host_descriptors)[vector];
        if(!threadIdx.x && (descriptor_words&1))device_descriptors[descriptor_words-1]=host_descriptors[descriptor_words-1];
    }
}
template<bool Publish,bool Cached=false,bool Gate=false,bool StableSnapshot=false,bool EarlyTags=false,int PlanItems=0>
__global__ void __launch_bounds__(64) prepare_warp(MfeFfnBatch b,
        const MfeInputQuantization* inputs,int count,int32_t* mapped_ids,uint32_t* ready,bool local_control,
        MfeCachedDispatch cache={},MfeExpertPlan expert_plan={}) {
    const int thread=threadIdx.x,lane=thread&31;
    __shared__ int cached_kinds[Cached?256:1];
    if(Publish && blockIdx.x==0) {
        const int publication_threads=b.tokens*b.routes<=64?32:64;
        if(thread<publication_threads) {
            for(int pair=thread;pair<b.tokens*b.routes;pair+=publication_threads)mapped_ids[pair]=b.ids[pair];
            __threadfence_system();__syncwarp();
        }
        if(publication_threads==64)__syncthreads();
        if(!thread) {
            ::cuda::atomic_ref<uint32_t,::cuda::thread_scope_system> publication(*ready);
            publication.store(1u,::cuda::std::memory_order_release);
        }
    }
    const auto flat=int64_t(blockIdx.x)*64+thread;
    const int slots=b.tokens*(b.routes+(b.shared?1:0));
    if(flat<int64_t(slots)*b.scale_stride)b.completion[flat]=0;
    if constexpr(PlanItems)if(blockIdx.x==0)build_expert_plan<PlanItems>(b,expert_plan);
    if(b.token_group_counts && blockIdx.x==0) {
        const int entries=b.tokens*b.routes,all_routes=b.routes+int(b.shared!=nullptr);
        if(b.token_group_buckets) {
            if(thread<4)b.token_group_buckets[thread*(b.token_groups+1)]=0;
            __syncthreads();
        }
        for(int entry=thread;entry<b.token_groups;entry+=64) {
            int length=0;
            if(entry<entries) {
                const int expert=b.ids[entry];
                if(unsigned(expert)<unsigned(b.experts)) {
                    int rank=0;
                    if constexpr(PlanItems)rank=expert_plan.pair_ranks[entry];
                    else for(int previous=0;previous<entry;++previous)rank+=b.ids[previous]==expert;
                    if constexpr(PlanItems) {
                        if(rank%b.token_group_capacity==0) {
                            length=min(b.token_group_capacity,expert_plan.expert_counts[expert]-rank);
                            const int first=expert_plan.expert_offsets[expert]+rank;
                            for(int i=0;i<length;++i) {
                                const int pair=expert_plan.sorted_pairs[first+i];
                                b.token_group_slots[entry*8+i]=(pair/b.routes)*all_routes+pair%b.routes;
                            }
                        }
                    }else if(rank%b.token_group_capacity==0)
                        for(int next=entry;next<entries && length<b.token_group_capacity;++next)
                            if(b.ids[next]==expert)b.token_group_slots[entry*8+length++]=
                                (next/b.routes)*all_routes+next%b.routes;
                }
            }else {
                const int first=(entry-entries)*b.token_group_capacity;
                for(int token=first;token<b.tokens && length<b.token_group_capacity;++token)
                    b.token_group_slots[entry*8+length++]=token*all_routes+b.routes;
            }
            b.token_group_counts[entry]=length;
            if(length && b.token_group_buckets) {
                const int bucket=length==1?0:length<=2?1:length<=4?2:3;
                auto* plan=b.token_group_buckets+bucket*(b.token_groups+1);
                const int index=atomicAdd(plan,1);plan[index+1]=entry;
            }
        }
    }
    if(b.plan_ready && blockIdx.x==0) {
        if constexpr(Cached) {
            __shared__ int selected[256];
            __shared__ int refresh[256];
            __shared__ uint64_t tags[256];
            const int entries=b.tokens*b.routes;
            auto* mapping=const_cast<int32_t*>(b.cohort_by_expert);
            bool needs_refresh=false;
            if constexpr(PlanItems) {
                for(int pair=thread;pair<entries;pair+=64) {
                    selected[pair]=0;
                    refresh[pair]=0;
                    tags[pair]=0;
                    cached_kinds[pair]=-2;
                }
                __syncthreads();
            }
            const int lookups=PlanItems?*expert_plan.unique_count:entries;
            for(int lookup=thread/4;lookup<lookups;lookup+=16) {
                const int pair=PlanItems?expert_plan.unique_pairs[lookup]:lookup;
                const int projection=thread&3;
                const unsigned mask=0xfu<<((thread&31)&~3);
                if constexpr(!PlanItems)
                    if(!projection) {
                        selected[pair]=0;
                        refresh[pair]=0;
                        tags[pair]=0;
                        cached_kinds[pair]=-2;
                    }
                const int expert=b.ids[pair];if(unsigned(expert)>=unsigned(b.experts))continue;
                bool duplicate=false;
                // The route's four lanes inspect disjoint prior positions.
                // Their vote preserves one writer per expert, including routes
                // repeated in another warp or an earlier loop iteration.
                if constexpr(!PlanItems)
                    for(int prior=projection;prior<pair;prior+=4)duplicate|=b.ids[prior]==expert;
                duplicate=__any_sync(mask,duplicate);
                if(duplicate)continue;
                // The snapshot publication below still protects this read.
                // Issue its PCIe request before the independent GPU mapping
                // reads, then consume the value only if a primary is missing.
                uint64_t early_tag=0;
                if constexpr(EarlyTags)if(projection==3)
                    early_tag=reinterpret_cast<const volatile MfeCachedExpert*>(cache.mapped)[expert].ready_mask;
                unsigned missing=0;
                // A relu-squared bank has no Gate cohort: projection 0 keeps its
                // empty primary layout and is never missing or refreshed.
                if(projection<3 && !(projection==0 && b.expert_activation==1)) {
                    const int p=projection;
                    const int pool=cache.primary_cohorts[p*b.experts+expert];
                    mapping[p*b.experts+expert]=pool;
                    if(unsigned(pool)>=unsigned(cache.primary_counts[p]))missing=1u<<p;
                    else {
                        const auto& d=b.projections[b.projection_offsets[p]+pool];
                        const int local=d.expert_local?d.expert_local[expert]:0;
                        if(local<0 || local>=d.local_experts)missing=1u<<p;
                    }
                }
                missing|=__shfl_xor_sync(mask,missing,1,4);
                missing|=__shfl_xor_sync(mask,missing,2,4);
                if constexpr(EarlyTags)early_tag=__shfl_sync(mask,early_tag,((thread&31)|3));
                if(projection)continue;
                unsigned remaining=missing;
                if(missing) {
                    const auto tag=EarlyTags?early_tag:
                        reinterpret_cast<const volatile MfeCachedExpert*>(cache.mapped)[expert].ready_mask;
                    const unsigned available=unsigned(tag&7)&missing;
                    remaining=missing&~available;
                    // Cached GU can run before the Down DMA, just like GU in
                    // the primary arena. Other partial bundles keep transfers.
                    if(!remaining || (remaining==4 && (available&3))) {
                        const auto previous=cache.snapshot_tags[expert];
                        const unsigned copied=(previous&~uint64_t(7))==(tag&~uint64_t(7))?unsigned(previous&7):0;
                        selected[pair]=int(available);refresh[pair]=int(available&~copied);
                        tags[pair]=(tag&~uint64_t(7))|copied|available;
                        needs_refresh|=refresh[pair]!=0;
                    }
                }
                const unsigned unresolved=selected[pair]?remaining:missing;
                cached_kinds[pair]=unresolved?(unresolved==4?6:2):1;
            }
            if constexpr(StableSnapshot)needs_refresh=__syncthreads_or(needs_refresh)!=0;
            else {__syncthreads();needs_refresh=true;}
            if(needs_refresh) {
              for(int vector=thread;vector<entries*12;vector+=64) {
                const int pair=vector/12,p=(vector/4)%3,f=(vector%4)*2;
                if(!(refresh[pair]&(1<<p)))continue;
                const int expert=b.ids[pair];
                auto& d=cache.projections[b.projection_offsets[p]+cache.primary_counts[p]+expert];
                const int fields=d.family==1?7:d.family==3?2:4;if(f>=fields)continue;
                const auto values=reinterpret_cast<const uint4*>(cache.mapped[expert].fields[p])[f/2];
                d.fields[f]=reinterpret_cast<const void*>(uint64_t(values.x)|(uint64_t(values.y)<<32));
                if(f+1<fields)d.fields[f+1]=reinterpret_cast<const void*>(uint64_t(values.z)|(uint64_t(values.w)<<32));
              }
              __syncthreads();
            }
            // No mapped validity or field reader remains after the refresh
            // barrier (or the needs_refresh reduction). CPU fetch may proceed
            // while the CTA finishes its independent GPU mappings and kinds.
            if constexpr(Publish)if(!thread) {
                ::cuda::atomic_ref<uint32_t,::cuda::thread_scope_system> publication(*cache.snapshot_ready);
                publication.store(1u,::cuda::std::memory_order_release);
            }
            for(int pair=thread;pair<entries;pair+=64)if(selected[pair]) {
                const int expert=b.ids[pair];
                cache.snapshot_tags[expert]=tags[pair];
                for(int p=0;p<3;++p)if(selected[pair]&(1<<p))
                    mapping[p*b.experts+expert]=cache.primary_counts[p]+expert;
            }
            // The cancellation barrier immediately below already joins every
            // mapping writer. Stable snapshots need no additional join here.
            if constexpr(!StableSnapshot)__syncthreads();
        }
        __shared__ uint32_t cancelled;
        if(thread==0) {
            // Resident work uses only validated GPU descriptors. The mandatory
            // transfer-ready kernel observes cancellation before consuming any
            // host-selected descriptors or combining the final result.
            if(local_control && b.resident_plan_overlap && b.transfer_ready)cancelled=0;
            else {
                const auto* host_abort=reinterpret_cast<const volatile uint32_t*>(b.host_aborted);
                if(!b.resident_plan_overlap)
                    while(!*reinterpret_cast<const volatile uint32_t*>(b.plan_ready) && !*host_abort)__nanosleep(64);
                cancelled=*host_abort?1:0;
            }
            *b.aborted=cancelled;
        }
        __syncthreads();
        if(b.resident_plan_overlap) {
            for(int pair=thread;pair<b.tokens*b.routes;pair+=64) {
                const int expert=b.ids[pair];if(unsigned(expert)>=unsigned(b.experts))continue;
                if constexpr(Cached) {
                    // The complete primary/cache test above already proved
                    // this classification. Duplicate routes have one writer.
                    if(cached_kinds[pair]>0)b.kinds[expert]=cancelled?-1:cached_kinds[pair];
                }else atomicExch(b.kinds+expert,cancelled?-1:residency_kind(b,expert));
            }
        }else for(int expert=thread;expert<b.experts;expert+=64) {
            b.kinds[expert]=cancelled?-1:reinterpret_cast<const volatile int32_t*>(b.host_kinds)[expert];
            b.transfer_index[expert]=cancelled?-1:reinterpret_cast<const volatile int32_t*>(b.host_transfer_index)[expert];
        }
    }
    // A warp owns one group, including GS33..64 via two values per lane.
    // Quantization needs no shared scratch or block-wide reduction barriers.
    int entry=int(blockIdx.x)*2+(thread>>5);const MfeInputQuantization* d=nullptr;
    if constexpr(Gate) {
        if(entry>=cache.gate.quantization_total)return;
        entry=(entry+cache.gate.quantization_rotation)%cache.gate.quantization_total;
    }
    for(int i=0;i<count;++i) {
        const int length=b.tokens*inputs[i].groups;
        if(entry<length){d=inputs+i;break;}entry-=length;
    }
    if(!d)return;
    const int token=entry/d->groups,group=entry%d->groups;
    float values[2]{};
#pragma unroll
    for(int part=0;part<2;++part) {
        const int local=lane+part*32,column=group*d->group_size+local;
        if(local<d->group_size && column<b.input_width)
            values[part]=__half2float(b.input[int64_t(token)*b.input_width+column]);
    }
    float maximum=fmaxf(fabsf(values[0]),fabsf(values[1]));
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
    const float scale=maximum>0.0f?maximum/127.0f:1.0f;
    if(!lane)d->scales[entry]=scale;
#pragma unroll
    for(int part=0;part<2;++part) {
        const int local=lane+part*32,column=group*d->group_size+local;
        if(local<d->group_size) {
            const int code=column<b.input_width?max(-127,min(127,int(roundf(values[part]/scale)))):0;
            d->values[int64_t(token)*d->groups*d->group_size+column]=int8_t(code);
        }
    }
    if constexpr(Gate)if(d->values==cache.gate.activation) {
        __syncwarp();
        if(b.tokens==1)prepare_shared_gate<true>(cache.gate,token,group);
        else prepare_shared_gate<false>(cache.gate,token,group);
    }
}
template<bool Publish,bool Cached=false,bool Gate=false,bool StableSnapshot=false,bool EarlyTags=false>
void launch_prepare(const MfeFfnBatch& b,const MfeInputQuantization* inputs,int count,
        int32_t* mapped_ids,uint32_t* ready,bool local_control,const MfeCachedDispatch& cache,
        const MfeExpertPlan& plan,int blocks,cudaStream_t stream) {
    if(plan.sorted_pairs && b.tokens*b.routes<=kMfeExpertPlanMaxPairs) {
        const int pairs=b.tokens*b.routes;
        if(pairs<=256)
            prepare_warp<Publish,Cached,Gate,StableSnapshot,EarlyTags,4><<<blocks,64,0,stream>>>(
                b,inputs,count,mapped_ids,ready,local_control,cache,plan);
        else if(pairs<=512)
            prepare_warp<Publish,Cached,Gate,StableSnapshot,EarlyTags,8><<<blocks,64,0,stream>>>(
                b,inputs,count,mapped_ids,ready,local_control,cache,plan);
        else
            prepare_warp<Publish,Cached,Gate,StableSnapshot,EarlyTags,16><<<blocks,64,0,stream>>>(
                b,inputs,count,mapped_ids,ready,local_control,cache,plan);
    }else prepare_warp<Publish,Cached,Gate,StableSnapshot,EarlyTags><<<blocks,64,0,stream>>>(
        b,inputs,count,mapped_ids,ready,local_control,cache);
}
}
void mfe_ffn_prepare_publish(const MfeFfnBatch& b,const MfeInputQuantization* inputs,int count,int groups,
        int32_t* mapped_ids,uint32_t* ready,cudaStream_t stream,const MfeExpertPlan& plan) {
    int blocks=(b.tokens*groups+1)/2;
    blocks=std::max(blocks,(b.tokens*(b.routes+(b.shared?1:0))*b.scale_stride+63)/64);
    const auto* control=std::getenv("MFQ_MFE_PREPARE_LOCAL_CONTROL");
    if(ready)launch_prepare<true>(b,inputs,count,mapped_ids,ready,!control || control[0]!='0',{},plan,blocks,stream);
    else launch_prepare<false>(b,inputs,count,nullptr,nullptr,!control || control[0]!='0',{},plan,blocks,stream);
}
void mfe_ffn_prepare_warp(const MfeFfnBatch& b,const MfeInputQuantization* inputs,int count,int groups,
        cudaStream_t stream,const MfeExpertPlan& plan) {
    mfe_ffn_prepare_publish(b,inputs,count,groups,nullptr,nullptr,stream,plan);
}
void mfe_ffn_prepare_publish_cached(const MfeFfnBatch& b,const MfeInputQuantization* inputs,int count,int groups,
        int32_t* mapped_ids,uint32_t* ready,const MfeCachedDispatch& cache,cudaStream_t stream,
        const MfeExpertPlan& plan) {
    static std::atomic<bool> cooperative_reported{false};
    if(!cooperative_reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
        "mfe_prepare_cooperative_dispatch scope=first_cached_prepare enabled=1 route_lanes=4 "
        "snapshot_publish=after_mapped_reads duplicate_check=%s tokens=%d\n",
        plan.sorted_pairs?"expert_plan":"quad",b.tokens);
    int blocks=std::max((b.tokens*groups+1)/2,
        (b.tokens*(b.routes+(b.shared?1:0))*b.scale_stride+63)/64);
    const auto* stable=std::getenv("MFQ_MFE_CACHE_PREPARE_SYNC");
    const auto* tag_overlap=std::getenv("MFQ_MFE_TAG_OVERLAP");
    if(cache.primary_incomplete && (!tag_overlap || tag_overlap[0]!='0') && (!stable || stable[0]!='0')) {
        if(cache.gate.activation)
            launch_prepare<true,true,true,true,true>(b,inputs,count,mapped_ids,ready,true,cache,plan,blocks,stream);
        else launch_prepare<true,true,false,true,true>(b,inputs,count,mapped_ids,ready,true,cache,plan,blocks,stream);
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed)) {
            std::fprintf(stderr,"mfe_prepare_tag_overlap_dispatch scope=first_cached_prepare enabled=1 tokens=%d gate=%d\n",
                b.tokens,int(cache.gate.activation!=nullptr));
            std::fprintf(stderr,"mfe_cache_prepare_sync_dispatch scope=first_cached_prepare enabled=1 tokens=%d gate=%d tag_overlap=1\n",
                b.tokens,int(cache.gate.activation!=nullptr));
        }
        return;
    }
    if(!stable || stable[0]!='0') {
        if(cache.gate.activation)
            launch_prepare<true,true,true,true>(b,inputs,count,mapped_ids,ready,true,cache,plan,blocks,stream);
        else launch_prepare<true,true,false,true>(b,inputs,count,mapped_ids,ready,true,cache,plan,blocks,stream);
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "mfe_cache_prepare_sync_dispatch scope=first_cached_prepare enabled=1 tokens=%d gate=%d\n",
            b.tokens,int(cache.gate.activation!=nullptr));
        return;
    }
    if(cache.gate.activation)
        launch_prepare<true,true,true>(b,inputs,count,mapped_ids,ready,true,cache,plan,blocks,stream);
    else launch_prepare<true,true>(b,inputs,count,mapped_ids,ready,true,cache,plan,blocks,stream);
}
void mfe_ffn_wait_transfer_packet(const MfeFfnBatch& b,const int32_t* packet,cudaStream_t stream) {
    transfer_packet<false><<<1,32,0,stream>>>(b,packet,nullptr,nullptr,0);
}
void mfe_ffn_wait_transfer_packet_snapshot(const MfeFfnBatch& b,const int32_t* packet,
        const void* table,void* snapshot,uint64_t capacity,cudaStream_t stream,
        const void* mapped_descriptors,void* device_descriptors,uint64_t descriptor_capacity_words,bool cached_resident_packet,
        int32_t* transfer_routes) {
    if(mapped_descriptors && device_descriptors) {
        if(cached_resident_packet) {
            const auto* early=std::getenv("MFQ_MFE_EARLY_GU");
            const bool early_gu=b.resident_plan_overlap && (!early || early[0]!='0');
            transfer_packet<true,true,true><<<1,128,0,stream>>>(b,packet,
                static_cast<const uint64_t*>(table),static_cast<uint64_t*>(snapshot),capacity,
                static_cast<const uint64_t*>(mapped_descriptors),static_cast<uint64_t*>(device_descriptors),descriptor_capacity_words,
                b.tokens==1 && b.routes<=32?transfer_routes:nullptr,early_gu);
            return;
        }
        transfer_packet<true,false,true><<<1,128,0,stream>>>(b,packet,
            static_cast<const uint64_t*>(table),static_cast<uint64_t*>(snapshot),capacity,
            static_cast<const uint64_t*>(mapped_descriptors),static_cast<uint64_t*>(device_descriptors),descriptor_capacity_words);
        return;
    }
    const auto* option=std::getenv("MFQ_MFE_DEVICE_RESIDENT_PACKET");
    if(b.tokens==1 && b.routes<=32 && b.resident_plan_overlap && option && option[0]!='0') {
        transfer_packet<true,true><<<1,32,0,stream>>>(b,packet,static_cast<const uint64_t*>(table),
            static_cast<uint64_t*>(snapshot),capacity);
        return;
    }
    transfer_packet<true><<<1,b.tokens<=1?32:128,0,stream>>>(b,packet,static_cast<const uint64_t*>(table),
        static_cast<uint64_t*>(snapshot),capacity);
}
}
