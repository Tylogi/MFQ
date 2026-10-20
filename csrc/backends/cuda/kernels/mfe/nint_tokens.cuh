#pragma once

// Register-only weight reuse for a group of tokens routed to one NINT expert.
namespace mfq::cuda {
namespace {
template<int GroupSize,int Capacity,int Rows=2>
__device__ __forceinline__ void nint_token_pair(const MfePackedProjection& d,int local,int first,
        const int8_t* const* inputs,const float* const* scales,int count,__half result[Capacity][4],int destination) {
    constexpr int chunks=GroupSize/4;
    const int lane=threadIdx.x;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0])+size_t(local)*d.q_expert_stride;
    const auto* bits=static_cast<const uint8_t*>(d.fields[1]);
    const auto* offsets=static_cast<const int64_t*>(d.fields[2]);
    const auto* subscale=static_cast<const uint8_t*>(d.fields[3]);
    const auto* submin=static_cast<const uint8_t*>(d.fields[4]);
    const auto* outer_scale=static_cast<const float*>(d.fields[5]);
    const auto* outer_min=static_cast<const float*>(d.fields[6]);
    float partial[Capacity][Rows]{};
    int widths[Rows]{};uint64_t starts[Rows]{};float factors[Rows]{},minima[Rows]{};
#pragma unroll
    for(int r=0;r<Rows;++r)if(first+r<d.output_rows) {
        const int neuron=local*d.output_rows+first+r;
        widths[r]=bits[neuron];starts[r]=uint64_t(offsets[neuron]);
        factors[r]=outer_scale[neuron];minima[r]=outer_min[neuron];
    }
    for(int group=lane;group<d.groups;group+=32) {
#pragma unroll
        for(int r=0;r<Rows;++r)if(first+r<d.output_rows) {
            int weight[chunks];
#pragma unroll
            for(int chunk=0;chunk<chunks;chunk+=2) {
                const auto offset=starts[r]+uint64_t(group*GroupSize+chunk*4)*widths[r];
                if(chunk+1<chunks) {
                    const auto packed=packed_nint::unpack_nint_codes8_packed(stream,offset,widths[r]);
                    const uint32_t low=uint32_t(packed),second=uint32_t(packed>>(4*widths[r]));
                    const uint32_t a=__byte_perm(low,low>>(2*widths[r]),0x5410);
                    const uint32_t b=__byte_perm(second,second>>(2*widths[r]),0x5410);
                    const uint32_t mask=((1u<<widths[r])-1u)*0x01010101u;
                    weight[chunk]=int(__byte_perm(a,a>>widths[r],0x6240)&mask);
                    weight[chunk+1]=int(__byte_perm(b,b>>widths[r],0x6240)&mask);
                }else weight[chunk]=packed_nint::unpack_nint_codes4(stream,offset,widths[r]);
            }
            const auto meta=int64_t(local*d.output_rows+first+r)*d.groups+group;
            const float scale=float(subscale[meta]),minimum=float(submin[meta]);
#pragma unroll
            for(int k=0;k<Capacity;++k)if(k<count) {
                int dot=0,sum=0;
#pragma unroll
                for(int chunk=0;chunk<chunks;++chunk) {
                    const int activation=packed_nint::load_i8x4(inputs[k]+group*GroupSize+chunk*4);
                    const int s=__dp4a(0x01010101,activation,0);sum+=s;
                    dot+=widths[r]==8?__dp4a(weight[chunk]^int(0x80808080u),activation,0)+128*s:
                        __dp4a(weight[chunk],activation,0);
                }
                partial[k][r]+=scales[k][group]*(factors[r]*scale*float(dot)-minima[r]*minimum*float(sum));
            }
        }
    }
#pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
#pragma unroll
        for(int r=0;r<Rows;++r) {
#pragma unroll
            for(int offset=16;offset>0;offset>>=1)partial[k][r]+=__shfl_xor_sync(0xffffffffu,partial[k][r],offset);
            result[k][destination+r]=__float2half(partial[k][r]);
        }
    }
}

}
}
