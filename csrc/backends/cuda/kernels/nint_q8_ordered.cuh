#pragma once
// GS48, one activation row. Four producer warps decode different groups,
// then warp 0 preserves the original chunk-lane accumulation and XOR order.
__global__ void __launch_bounds__(128) nint_q8_ordered_kernel(
        const uint8_t* __restrict__ bitstream,const int64_t* __restrict__ offsets,
        const uint8_t* __restrict__ subgroup_scale,const uint8_t* __restrict__ subgroup_minimum,
        const float* __restrict__ neuron_scale,const float* __restrict__ neuron_minimum,
        const int8_t* __restrict__ activation,const float* __restrict__ activation_scale,
        __half* __restrict__ output,int rows,int groups) {
    const int row=blockIdx.x,lane=threadIdx.x,warp=threadIdx.y;
    if(row>=rows)return;
    const int relative_group=lane/12,chunk=lane%12;
    const auto* weights=bitstream+(uint64_t(offsets[row])>>3);
    const auto* scales=subgroup_scale+size_t(row)*groups;
    const auto* minima=subgroup_minimum+size_t(row)*groups;
    const float outer_scale=neuron_scale[row],outer_minimum=neuron_minimum[row];
    __shared__ float terms[64][12];
    if(relative_group<2)for(int group=warp*2+relative_group;group<groups;group+=8) {
        const int column=group*48+chunk*4;
        const int w=*reinterpret_cast<const int*>(weights+column);
        const int x=*reinterpret_cast<const int*>(activation+column);
        const int sum=__dp4a(0x01010101,x,0);
        const int dot=__dp4a(w^int(0x80808080u),x,0)+128*sum;
        // Store before multiplying by activation_scale. The production loop
        // contracts that multiplication with the accumulator addition.
        terms[group][chunk]=outer_scale*float(scales[group])*float(dot)
            -outer_minimum*float(minima[group])*float(sum);
    }
    __syncthreads();
    if(warp!=0)return;
    float accumulator=0.0f;
    if(relative_group<2)for(int group=relative_group;group<groups;group+=2)
        accumulator+=activation_scale[group]*terms[group][chunk];
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)
        accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,offset);
    if(lane==0)output[row]=__float2half_rn(accumulator);
}
