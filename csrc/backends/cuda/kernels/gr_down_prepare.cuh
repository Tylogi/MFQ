#pragma once

// Every replay on a stream is ordered. CUDA initializes the counters once;
// the last Down CTA resets its counter before the next operation can start.
__device__ unsigned gr_down_prepare_counters[GrNormCounterSlots*7]{};

// A 64-row MMA tile overlaps at most two GS48 activation groups. Publish
// only actual result writers, then finish each group in its last producer.
__device__ __forceinline__ void gr_down_mma_finish_up_prepare(
        const float* parts,int splits,int tile,bool half,GrTwoPlane q,int counter_slot) {
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int first_row=tile*64,last_row=min(first_row+64,320)-1;
    const int first_group=first_row/48,last_group=last_row/48;
    __shared__ bool completed[2];
    if((lane&3)==0)__threadfence();
    __syncthreads();
    if(tid<2) {
        completed[tid]=false;
        const int group=first_group+tid;
        if(group<=last_group) {
            const int begin=group*48,end=min(begin+48,320)-1;
            const unsigned producers=unsigned((end/64-begin/64+1)*splits);
            auto* counter=gr_down_prepare_counters+counter_slot*7+group;
            completed[tid]=atomicAdd(counter,1u)==producers-1;
        }
    }
    __syncthreads();
    if(warp<2 && completed[warp]) {
        const int group=first_group+warp,begin=group*48,width=min(48,320-begin);
        const auto activate=[&](int column) {
            if(column>=width)return 0.f;
            const float value=gr_reduce_split_major_parts<true>(parts,0,begin+column,320,splits);
            return mhc_activated(value,half,4,false);
        };
        q.high+=2*begin;q.scales+=group;q.sums+=group;
        MhcMatrix matrix{};matrix.width=width;matrix.groups=1;
        gr_quantize_activation_values<48,false,true,true,true>(matrix,activate(lane),
            lane<16?activate(lane+32):0.f,q,0,0,lane);
        if(lane==0)atomicExch(gr_down_prepare_counters+counter_slot*7+group,0u);
    }
}

template<int Threads,bool Packed16=false,bool VectorParts=false,bool SplitMajor=false,bool PairedStats=false,bool MmaPlanes=false,bool HalfUp=false>
__device__ __forceinline__ void gr_down_finish_up_prepare(
        const float* parts,int down_splits,int row_tile,int first_row,bool half,GrTwoPlane q,int counter_slot) {
    static_assert(Threads==256);
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int group=first_row/48,begin=group*48,width=min(48,320-begin);
    const unsigned participants=((width+row_tile-1)/row_tile)*down_splits;
    __shared__ bool last;
    // Parts are written by different lanes. Every writer publishes before
    // thread zero joins the grid's device-scope completion chain.
    if(lane==0)__threadfence();
    __syncthreads();
    auto* counter=gr_down_prepare_counters+counter_slot*7+group;
    if(tid==0) {
        last=atomicAdd(counter,1u)==participants-1u;
    }
    __syncthreads();
    if(!last)return;
    const volatile float* visible_parts=parts;
    if constexpr(Packed16) {
        if(warp==0) {
            const auto activate=[&](int column) {
                if(column>=width)return 0.f;
                const int row=begin+column;
                float sum;
                if constexpr(SplitMajor)sum=gr_reduce_split_major_parts<true>(parts,0,row,320,down_splits);
                else if constexpr(VectorParts)sum=gr_reduce_down_parts<true>(parts,0,row,down_splits);
                else {
                    sum=down_splits==1?visible_parts[row]:0.f;
                    if(down_splits>1)for(int split=0;split<down_splits;++split)
                        sum+=visible_parts[row*down_splits+split];
                }
                return mhc_activated(sum,half,4,false);
            };
            q.high+=MmaPlanes?192*group:2*begin;q.scales+=group;q.sums+=group;
            MhcMatrix tile{};tile.width=width;tile.groups=1;
            if constexpr(HalfUp) {
                // Half activations are already rounded by mhc_activated.
                // Prepare the exact half values and affine group sum in the
                // existing last Down CTA, avoiding a separate launch.
                gr_prepare_up_half_values<true>(activate(lane),lane<16?activate(lane+32):0.f,
                    {reinterpret_cast<__half*>(q.high),nullptr,q.scales},0,0,lane);
            }else gr_quantize_activation_values<48,false,true,true,PairedStats,MmaPlanes>(tile,activate(lane),
                lane<16?activate(lane+32):0.f,q,0,0,lane);
            if(lane==0)atomicExch(counter,0u);
        }
        return;
    }
    __shared__ float low[48];
    if(tid<width) {
        const int row=begin+tid;
        float sum=down_splits==1?visible_parts[row]:0.f;
        if(down_splits>1)for(int split=0;split<down_splits;++split)
            sum+=visible_parts[row*down_splits+split];
        low[tid]=mhc_activated(sum,half,4,false);
    }
    __syncthreads();
    // Row tiles divide GS48. Independent group counters avoid serializing
    // the entire Down grid on one atomic; the split sum order is unchanged.
    if(warp==0) {
        q.high+=begin;q.low+=begin;q.scales+=group;q.sums+=group;
        MhcMatrix tile{};tile.width=width;tile.groups=1;
        gr_quantize_activation_group<48,false,true>(tile,low,false,q,0,0,lane);
    }
    __syncthreads();
    if(tid==0) {
        atomicExch(counter,0u);
    }
}
