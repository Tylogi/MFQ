// FP32 batched GR projection from compressed NINT weights. The only decoded
// weights are a small shared-memory tile reused by sixty-four token rows.
namespace {
bool gr_prefill_selected(const mfq_tensor_backend::Tensor& input) {
    const auto* option=std::getenv("MFQ_GR_PREFILL_MATMUL");
    return (!option || std::atoi(option)!=0) && input.dim()>0 && input.size(-1)>0 &&
        input.numel()/input.size(-1)>=32;
}

template<class Input,bool AlignedQ8,bool Zero,int BM=64,int RegisterN=4,int BK=32,int RegisterM=2,bool CompactK=false>
__global__ __launch_bounds__(BM*32/(RegisterN*RegisterM),1) void gr_prefill_matmul_kernel(const Input* input,
        const uint8_t* packed,const uint8_t* row_bits,const int64_t* row_offsets,
        const uint8_t* scales,const uint8_t* minima,const float* row_scales,
        const float* row_minima,const __half* zero_scales,float* output,
        int rows,int width,int outputs,int groups,int gs) {
    constexpr int BN=32,Pad=BK+1,Threads=BM*32/(RegisterN*RegisterM),Columns=BN/RegisterN;
    static_assert(BK%4==0,"GR K tile must preserve the four FP32 accumulator streams");
    __shared__ float a[BM][Pad],w[BN][Pad];
    const int tid=threadIdx.x,m=tid/Columns,n=(tid%Columns)*RegisterN;
    const int row0=int(blockIdx.y)*BM,out0=int(blockIdx.x)*BN;
    float acc[RegisterM][RegisterN][4]{};
    for(int k0=0;k0<width;k0+=BK) {
        for(int i=tid;i<BM*BK;i+=Threads) {
            const int r=i/BK,c=i%BK;
            a[r][c]=row0+r<rows && k0+c<width ? float(input[(int64_t(row0+r)*width)+k0+c]) : 0;
        }
        for(int i=tid;i<BN*BK;i+=Threads) {
            const int r=i/BK,c=i%BK,neuron=out0+r,column=k0+c;
            float value=0;
            if(neuron<outputs && column<width) {
                const int64_t meta=int64_t(neuron)*groups+column/gs;
                float q,scale,minimum;
                if constexpr(Zero) {
                    q=float(static_cast<int8_t>(packed[meta*gs+column%gs]));
                    scale=__half2float(zero_scales[meta]);minimum=0;
                } else {
                    const int bits=AlignedQ8?8:row_bits[neuron];
                    const uint64_t bit=uint64_t(row_offsets[neuron])+uint64_t(column)*bits;
                    unsigned word=packed[bit>>3];const int shift=int(bit&7);
                    if(shift+bits>8)word|=unsigned(packed[(bit>>3)+1])<<8;
                    q=float((word>>shift)&((1u<<bits)-1));
                    scale=row_scales[neuron]*float(scales[meta]);
                    minimum=row_minima[neuron]*float(minima[meta]);
                }
                value=__fsub_rn(__fmul_rn(scale,q),minimum);
            }
            w[r][c]=value;
        }
        __syncthreads();
        if constexpr(CompactK) {
        // Explicit constant accumulator slots keep the arrays in registers;
        // only four K positions are live rather than the entire shared tile.
        #pragma unroll 1
        for(int kb=0;kb<BK;kb+=4) {
            #pragma unroll
            for(int slot=0;slot<4;++slot) {
                #pragma unroll
                for(int r=0;r<RegisterM;++r) {
                    const float x=a[m+r*(BM/RegisterM)][kb+slot];
                    #pragma unroll
                    for(int j=0;j<RegisterN;++j)
                        acc[r][j][slot]=fmaf(x,w[n+j][kb+slot],acc[r][j][slot]);
                }
            }
        }
        } else {
        #pragma unroll
        for(int k=0;k<BK;++k) {
            #pragma unroll
            for(int r=0;r<RegisterM;++r) {
                const float x=a[m+r*(BM/RegisterM)][k];
                #pragma unroll
                for(int j=0;j<RegisterN;++j)acc[r][j][k&3]=fmaf(x,w[n+j][k],acc[r][j][k&3]);
            }
        }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<RegisterM;++r)if(row0+m+r*(BM/RegisterM)<rows) {
        #pragma unroll
        for(int j=0;j<RegisterN;++j)if(out0+n+j<outputs)
            output[int64_t(row0+m+r*(BM/RegisterM))*outputs+out0+n+j]=
                (acc[r][j][0]+acc[r][j][1])+(acc[r][j][2]+acc[r][j][3]);
    }
}

mfq_tensor_backend::Tensor gr_prefill_matmul(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input) {
    namespace tb=mfq_tensor_backend;
    auto x=input.scalar_type()==tb::kFloat16 ? input.contiguous() : input.to(tb::kFloat32).contiguous();
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.size(-1)==weight.neuron_len && weight.q_packed.is_cuda() &&
        input.device()==weight.q_packed.device() && weight.gs>0 && weight.neuron_len>0 &&
        weight.neuron_len<=std::numeric_limits<int>::max() && weight.out>0 && weight.out<=std::numeric_limits<int>::max() &&
        weight.ng==(weight.neuron_len+weight.gs-1)/weight.gs,
        "GR prefill projection geometry mismatch");
    const auto rows=x.numel()/weight.neuron_len;
    MFQ_RUNTIME_CHECK(rows<=65535*16,"GR prefill projection exceeds CUDA grid geometry");
    MfqCudaGuard guard(input.device());
    const auto* option=std::getenv("MFQ_GR_PREFILL_MATMUL");
    int major=0;
    const bool requested_tf32=option && std::atoi(option)==2;
    const bool requested_half2=option && std::atoi(option)==3;
    int mode=option?std::atoi(option):17;
    const bool compact_k=mode==19 || mode==20;
    if(compact_k)mode=mode==19?6:17;
    if(mode==16 || mode==17 || mode==18) {
        const bool expansion=weight.out>=weight.neuron_len;
        mode=expansion?(mode>=17?(x.scalar_type()==tb::kFloat32?14:(mode==18?15:9)):(x.scalar_type()==tb::kFloat32?5:9))
            :(rows>=2048?9:rows>=128?(mode==17?11:13):1);
    }
    if(mode==12)mode=weight.out>=weight.neuron_len?(x.scalar_type()==tb::kFloat32?5:9):(rows>=2048?9:rows>=128?11:1);
    if(mode==6)mode=weight.out>=weight.neuron_len?(x.scalar_type()==tb::kFloat32?5:9):(rows>=2048?9:1);
    const int exact_tile=mode==4?128:mode==5?256:64;
    if(requested_tf32 || requested_half2)MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,input.get_device()));
    const bool tf32=requested_tf32 && major>=8;
    const bool half2=requested_half2 && major>=7;
    auto shape=input.sizes().vec();shape.back()=weight.out;
    auto output=tb::empty(shape,input.options().dtype(tb::kFloat32));
    const auto typed=[&](auto tag,auto aligned_tag,auto zero_tag) {
        using Input=decltype(tag);constexpr bool aligned=decltype(aligned_tag)::value,zero=decltype(zero_tag)::value;
        const auto launch_kernel=[&](auto kernel,int tile_m,int threads) {
        kernel<<<dim3(unsigned((weight.out+31)/32),unsigned((rows+tile_m-1)/tile_m)),threads,0,mfq_current_cuda_stream()>>>(
            x.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),
            zero?nullptr:weight.row_q_bits.data_ptr<uint8_t>(),zero?nullptr:weight.row_q_bit_offsets.data_ptr<int64_t>(),
            zero?nullptr:weight.sub_scale.data_ptr<uint8_t>(),zero?nullptr:weight.sub_min.data_ptr<uint8_t>(),
            zero?nullptr:weight.neuron_scale.data_ptr<float>(),zero?nullptr:weight.neuron_min.data_ptr<float>(),
            zero?weight.q8_zero_scale.data_ptr<__half>():nullptr,output.data_ptr<float>(),int(rows),
            int(weight.neuron_len),int(weight.out),int(weight.ng),int(weight.gs));
        };
        if(compact_k) {
            if(mode==14)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,4,true>,256,512);
            else if(mode==11)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64,2,true>,128,512);
            else if(mode==9)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8,32,2,true>,256,512);
            else if(mode==5)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,2,true>,256,1024);
            else launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,4,32,2,true>,64,256);
            return;
        }
        if(half2)launch_kernel(gr_prefill_half2_kernel<Input,aligned,zero>,64,256);
        else if(tf32)launch_kernel(gr_prefill_tf32_kernel<Input,aligned,zero>,32,128);
        else if(mode==10)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,4,64>,64,256);
        else if(mode==11)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64>,128,512);
        else if(mode==13)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64,4>,128,256);
        else if(mode==14)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,4>,256,512);
        else if(mode==15)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8,32,4>,256,256);
        else if(mode==9)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8>,256,512);
        else if(mode==8)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,8>,128,256);
        else if(mode==7)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,8>,64,128);
        else if(exact_tile==256)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256>,256,1024);
        else if(exact_tile==128)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128>,128,512);
        else launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero>,64,256);
    };
    const auto launch=[&](auto tag) {
        if(weight.q8_zero)typed(tag,std::false_type{},std::true_type{});
        else if(weight.aligned_q8)typed(tag,std::true_type{},std::false_type{});
        else typed(tag,std::false_type{},std::false_type{});
    };
    if(x.scalar_type()==tb::kFloat16)launch(__half{});else launch(float{});
    MFQ_CUDA_CHECK(cudaGetLastError());return output;
}
} // namespace
