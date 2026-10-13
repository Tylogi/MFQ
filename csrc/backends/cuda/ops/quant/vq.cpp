#include "ops/execution/execution_options.h"
#include "kernels/norm_ops.h"
#include "kernels/quant_ops.h"
#include "vq.h"
#include "ops/execution/cuda_execution.h"
#include "ops/format.h"
#include "nvq_codebooks.generated.h"
#include "quant_dot.h"
#include "nvq_group.h"
#include "cpu_projection_rows.h"
#include "nvq_dense_pack.h"

bool prepare_nvq_dense_groups(NvqWeight& w) {
    if(w.dense_groups)return true;
    if(!w.indices_packed.is_cpu() || w.gs!=24 || w.sub_bits!=4 ||
            w.out<=0 || w.out>std::numeric_limits<int>::max() ||
            w.neuron_len<=0 || w.neuron_len>std::numeric_limits<int>::max() ||
            !mfq::cpu::nvq_dense_rows_dot_kernel(int(w.kernel_format)))return false;
    const int rows=int(w.out),pairs=int((w.neuron_len+7)/8),groups=int(w.ng);
    const bool d4=w.kernel_format==10 || w.kernel_format==11 || w.kernel_format==12 || w.kernel_format==15;
    if(d4 && (w.neuron_len+3)/4!=2*pairs)return false;
    // Every expert in an arena starts at a byte boundary. No per-expert
    // padding is inserted; unsupported odd shapes keep canonical storage.
    const int bits=w.kernel_format==12?9:(w.kernel_format==13 || w.kernel_format==15)?10:w.kernel_format==14?12:8;
    if((int64_t(rows)*(int64_t(pairs)*((d4?2:1)*bits+7)+int64_t(groups)*4))&7)return false;
    std::vector<uint8_t> compact;
#define PACK(B,P) mfq::cpu::dense::pack<B,P>(w.indices_packed.data_ptr<uint8_t>(),w.indices_packed.numel(), \
        w.aux_packed.data_ptr<uint8_t>(),w.aux_packed.numel(),w.sub_scale_packed.data_ptr<uint8_t>(), \
        w.sub_scale_packed.numel(),rows,pairs,groups)
    switch(w.kernel_format) {
        case 5:compact=PACK(8,1);break;case 13:compact=PACK(10,1);break;case 14:compact=PACK(12,1);break;
        case 10:case 11:compact=PACK(8,2);break;case 12:compact=PACK(9,2);break;case 15:compact=PACK(10,2);break;
        default:return false;
    }
#undef PACK
    if(compact.size()>size_t(w.indices_packed.numel()+w.aux_packed.numel()+w.sub_scale_packed.numel()))
        throw std::runtime_error("compact NVQ unexpectedly expanded weights");
    w.indices_packed=mfq_tensor_backend::tensor(compact);
    w.aux_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
    w.sub_scale_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
    w.dense_groups=true;
    return true;
}

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

void prepare_nvq_integer_codebook(NvqWeight& w) {
    const auto enabled=mfq::cuda::runtime_options::nvq1_integer_delta();
    if(!enabled || !*enabled || w.integer_codebook.defined() ||
       (w.kernel_format!=1 && w.kernel_format!=8) || !w.codebook.is_cuda())return;
    const auto canonical=w.codebook.to(mfq_tensor_backend::kCPU).contiguous();
    const bool small=w.kernel_format==8;
    const int bank_bytes=small?512*8:2048*8;
    if(canonical.numel()!=int64_t(small?2:1)*bank_bytes)
        throw std::invalid_argument("NVQ1 integer decode book has invalid canonical size");
    std::vector<int8_t> digits(2*bank_bytes);
    const auto* source=canonical.data_ptr<int8_t>();
    for(int negative=0;negative<2;++negative)for(int i=0;i<bank_bytes;++i) {
        const int v=source[(small?negative*bank_bytes:0)+i];
        if(v<-1 || v>1)throw std::invalid_argument("NVQ1 integer decode digits require -1/0/1");
        digits[negative*bank_bytes+i]=int8_t((small?32:8)*v+(negative?-1:1)*(small?5:1));
    }
    w.integer_codebook=mfq_tensor_backend::tensor(digits).to(w.codebook.device()).contiguous();
}
void prepare_nvq_decode_records(NvqWeight& w,bool /*immutable*/) {
    const auto enabled=mfq::cuda::runtime_options::nvq1_group_records();
    if(!enabled || !*enabled || w.decode_records.defined() ||
       (w.kernel_format!=1 && w.kernel_format!=8) || !w.indices_packed.is_cuda() || w.gs!=24 ||
       w.sub_bits!=(w.kernel_format==8?4:3))return;
    if(w.aux_packed.numel()==0){w.decode_records=w.indices_packed;return;}
    const auto indices=w.indices_packed.to(mfq_tensor_backend::kCPU).contiguous();
    const auto auxiliary=w.aux_packed.to(mfq_tensor_backend::kCPU).contiguous();
    const auto states=w.sub_scale_packed.to(mfq_tensor_backend::kCPU).contiguous();
    const auto bits=[](const mfq_tensor_backend::Tensor& data,int64_t bit,int width) {
        uint32_t value=0;
        for(int i=0;i<width;++i)if((bit+i)/8<data.numel())
            value|=uint32_t((data.data_ptr<uint8_t>()[(bit+i)/8]>>((bit+i)&7))&1u)<<i;
        return value;
    };
    const bool small=w.kernel_format==8;
    const int index_bits=small?9:11,record_bytes=small?4:5,nvec=int((w.neuron_len+7)/8);
    std::vector<uint8_t> records(size_t(w.out)*w.ng*record_bytes);
    for(int64_t row=0;row<w.out;++row)for(int group=0;group<w.ng;++group) {
        const int64_t linear=row*w.ng+group;
        uint64_t record=0;
        for(int segment=0;segment<3;++segment)if(group*3+segment<nvec)
            record|=uint64_t(bits(indices,(row*nvec+group*3+segment)*index_bits,index_bits))<<(segment*index_bits);
        record|=uint64_t(bits(states,linear*w.sub_bits,int(w.sub_bits)))<<(3*index_bits);
        record|=uint64_t(bits(auxiliary,linear,1))<<(3*index_bits+w.sub_bits);
        std::memcpy(records.data()+linear*record_bytes,&record,record_bytes);
    }
    w.decode_records=mfq_tensor_backend::tensor(records).to(w.indices_packed.device()).contiguous();
    w.indices_packed=w.decode_records;
    w.aux_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
    w.sub_scale_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
}

enum class NvqMatmulPath {
    Gemv,
    Mmq,
    OnlineF16,
    DequantGemm,
};

static NvqMatmulPath select_nvq_matmul_path(const NvqWeight & w, int M) {
    // Small expert matrices underfill the multi-column GEMV on SM86.
    // The vector decoder plus FP32-accumulating GEMM is faster from M=9.
    if (M >= 9 && M <= 16 && w.out <= 4096 && w.neuron_len <= 4096 &&
            !mfq::cuda::runtime_options::nvq_dense_reference()) {
        thread_local int cached_device = -1, major = 0, minor = 0;
        const int device = w.indices_packed.get_device();
        if (device != cached_device) {
            MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
            MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
            cached_device = device;
        }
        if (major == 8 && minor == 6) return NvqMatmulPath::DequantGemm;
    }
    const bool e8_family =
        w.format == 2 || w.format == 5 || w.format == 7 ||
        w.format == 8 || w.format == 9 ||
        w.format == 13 || w.format == 14;

    if (w.format == 8 && M >= 14 && M <= 16 && w.neuron_len >= 2 * w.out) {
        return NvqMatmulPath::DequantGemm;
    }

    if (w.format == 9 && M >= 13) {
        const bool wide_output = w.out * 8 >= w.neuron_len * 21;
        const bool wide_mmq =
            w.out >= 4096 && (w.out >= 2 * w.neuron_len || w.neuron_len >= 2 * w.out);
        if (M <= 15) {
            if (wide_mmq) return NvqMatmulPath::Mmq;
            return w.out >= 1024 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
        }
        if (M == 16) {
            if (w.out >= 4096) return NvqMatmulPath::Mmq;
            return w.out >= 1024 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
        }
        if (M <= 31) {
            if (wide_output) return NvqMatmulPath::OnlineF16;
            if (w.out >= 4096 && w.out >= w.neuron_len) return NvqMatmulPath::Mmq;
            return NvqMatmulPath::DequantGemm;
        }
        if (M == 32) {
            return w.out >= 4096 ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        if (M <= 47) {
            return wide_output ? NvqMatmulPath::OnlineF16 : NvqMatmulPath::DequantGemm;
        }
        if (M == 48) {
            return wide_output ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        if (M <= 63) {
            return wide_output ? NvqMatmulPath::OnlineF16 : NvqMatmulPath::DequantGemm;
        }
        if (M == 64) {
            if (wide_output) return NvqMatmulPath::OnlineF16;
            return w.out >= 8192 ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        return NvqMatmulPath::DequantGemm;
    }

    if (M <= 13) return NvqMatmulPath::Gemv;
    if (M == 14) {
        return w.out >= 2048 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
    }
    if (M == 15) {
        if (e8_family && w.out >= 8192) return NvqMatmulPath::Mmq;
        if (e8_family && w.out >= 2048) return NvqMatmulPath::Gemv;
        return NvqMatmulPath::DequantGemm;
    }
    if (M == 16) {
        if (e8_family && w.out >= 6144) return NvqMatmulPath::Mmq;
        if (e8_family && w.out >= 2048) return NvqMatmulPath::Gemv;
        if ((w.format == 3 || w.format == 10 ||
             w.format == 12 || w.format == 15) &&
            w.out >= 4096 && w.neuron_len >= 8192) {
            return NvqMatmulPath::Mmq;
        }
        return NvqMatmulPath::DequantGemm;
    }

    const bool wide_expansion = e8_family && w.out >= 3 * w.neuron_len;
    if (wide_expansion && ((M >= 17 && M <= 31) || (M >= 33 && M <= 47))) {
        return NvqMatmulPath::OnlineF16;
    }
    if (M == 32 && e8_family && w.out >= 8192) return NvqMatmulPath::Mmq;
    if (M == 48 && e8_family && w.out >= 12288) return NvqMatmulPath::Mmq;
    if (M == 64 && (w.format == 7 || w.format == 9) && w.out >= 8192) {
        return NvqMatmulPath::Mmq;
    }
    return NvqMatmulPath::DequantGemm;
}
static bool cpu_nvq_d4(int format) {
    return format == 3 || format == 10 || format == 11 ||
        format == 12 || format == 15;
}

static const int8_t * cpu_nvq_codebook(
        const int8_t * metadata,
        int format,
        uint32_t state) {
    constexpr int64_t header = 64;
    if (format == 1 || format == 2 || format == 3 || format == 8) {
        return metadata;
    }
    const uint32_t bank = format == 11
        ? state & 1u
        : static_cast<uint8_t>(metadata[36 + state]);
    if (format == 5) return metadata + header + bank * 256 * 8;
    if (format == 10 || format == 11) {
        return metadata + header + bank * 256 * 4;
    }
    if (format == 12) return metadata + header + bank * 512 * 4;
    if (format == 13) return metadata + header + bank * 1024 * 8;
    if (format == 14) return metadata + header + bank * 4096 * 8;
    if (format == 15) return metadata + header + bank * 1024 * 4;
    throw std::runtime_error("unsupported CPU NVQ codebook format");
}

static float cpu_nvq_scale(
        const int8_t * metadata,
        int format,
        float anchor,
        uint32_t state) {
    if (format == 1) {
        return anchor * static_cast<float>(state) * 0.125f;
    }
    if (format == 8) {
        return anchor * static_cast<float>(state) * 0.03125f;
    }
    if (format == 2 || format == 3) {
        return anchor * static_cast<float>(state);
    }
    if (format == 11) {
        return anchor * static_cast<float>((state >> 1) + 1);
    }
    const int64_t lut_offset = (format == 7 || format == 9) ? 8 : 4;
    return anchor * cpu_half_from_bytes(
        metadata, lut_offset + static_cast<int64_t>(state) * 2);
}

static int cpu_nvq_parity7(uint32_t value) {
    value &= 0x7fu;
    value ^= value >> 4;
    value ^= value >> 2;
    value ^= value >> 1;
    return static_cast<int>(value & 1u);
}

using CpuNvqDecodeView=mfq::cpu::NvqDecodeView;

static inline uint32_t cpu_nvq_bits(const uint8_t* data, int64_t bytes,
        int64_t bit, int bits) {
    const int64_t offset=bit>>3;
    uint32_t word=0;
    // Native canonical payloads have no mandatory tail padding.
    if (bytes-offset>=4) std::memcpy(&word,data+offset,4);
    else std::memcpy(&word,data+offset,static_cast<size_t>(bytes-offset));
    return (word>>(bit&7))&((1u<<bits)-1u);
}

static inline uint32_t cpu_nvq_state(const uint8_t* indices,const uint8_t* states,
        int64_t state_bytes,int64_t linear,int bits,int format) {
    if(state_bytes==0 && (format==1 || format==8)) {
        uint64_t record=0;const int bytes=format==8?4:5;
        std::memcpy(&record,indices+linear*bytes,bytes);
        return uint32_t(record>>(format==8?27:33))&((1u<<bits)-1u);
    }
    return cpu_nvq_bits(states,state_bytes,linear*bits,bits);
}

// Geometry, storage pointers and state tables are prepared once per GEMV.
// E8 indices/signs are consumed once per eight values, D4 once per four.
template<int Format>
static void cpu_decode_nvq_group(const CpuNvqDecodeView& w,
        int64_t row, int group, uint32_t state, int8_t values[24]) {
    constexpr bool d4 = Format==3 || Format==10 || Format==11 || Format==12 || Format==15;
    constexpr bool delta_format = Format==1 || Format==8;
    constexpr int bits = Format==1 ? 11 : Format==7 ? 7 :
        (Format==8 || Format==12) ? 9 : Format==9 ? 6 :
        (Format==13 || Format==15) ? 10 : Format==14 ? 12 : 8;
    const int8_t* bank=w.banks[state];
    int delta=1;uint64_t record=0;
    if constexpr(delta_format)if(w.aux_bytes==0)
        std::memcpy(&record,w.indices+(row*w.groups+group)*(Format==8?4:5),Format==8?4:5);
    if constexpr (delta_format) {
        const bool negative=w.aux_bytes==0?((record>>(Format==8?31:36))&1u)!=0:
            cpu_nvq_bits(w.aux,w.aux_bytes,row*w.groups+group,1)!=0;
        delta=negative ? -1 : 1;
        if constexpr (Format==8) bank+=int(negative)*512*8;
    }
    for (int chunk=0; chunk<3; ++chunk) {
        const int vector8=group*3+chunk;
        const int vector=d4 ? vector8*2 : vector8;
        if (vector>=w.nvec) {
            std::fill(values+chunk*8,values+chunk*8+8,int8_t(0));
            continue;
        }
        const int64_t linear=row*w.nvec+vector;
        const uint32_t code=delta_format && w.aux_bytes==0?
            uint32_t(record>>(chunk*bits))&((1u<<bits)-1u):bits==8 ? w.indices[linear] :
            cpu_nvq_bits(w.indices,w.index_bytes,linear*bits,bits);
        int8_t decoded[8];
        if constexpr (Format==7) {
            std::memcpy(decoded,bank+(code&7u)*4,4);
            // The second codebook begins after all eight first-half banks.
            std::memcpy(decoded+4,bank+256+state*32+(code>>3)*4,4);
        } else if constexpr (d4) {
            std::memcpy(decoded,bank+int64_t(code)*4,4);
            if (vector+1<w.nvec) {
                const uint32_t second=bits==8 ? w.indices[linear+1] :
                    cpu_nvq_bits(w.indices,w.index_bytes,(linear+1)*bits,bits);
                std::memcpy(decoded+4,bank+int64_t(second)*4,4);
            } else std::fill(decoded+4,decoded+8,int8_t(0));
        } else std::memcpy(decoded,bank+int64_t(code)*8,8);
        uint32_t mask8=0;
        if constexpr (!delta_format && Format!=7 && Format!=9) {
            const uint32_t mask7=cpu_nvq_bits(w.aux,w.aux_bytes,
                (row*w.nsign+vector8)*7,7);
            const uint32_t parity=cpu_nvq_parity7(mask7) ^
                ((Format==2 && w.sign_mode!=0) ? ((code>>7)&1u) : 0u);
            mask8=mask7|(parity<<7);
        }
        for (int i=0; i<8; ++i) {
            int value=decoded[i];
            if constexpr (Format==1) value=8*value+delta;
            else if constexpr (Format==8) value=32*value+5*delta;
            else value*=1-2*int((mask8>>i)&1u);
            values[chunk*8+i]=static_cast<int8_t>(value);
        }
    }
}

using CpuNvqDecoder=void(*)(const CpuNvqDecodeView&,int64_t,int,uint32_t,int8_t*);
static CpuNvqDecoder cpu_nvq_decoder(int format) {
    switch (format) {
#define MFQ_NVQ_CPU_CASE(F) case F: return cpu_decode_nvq_group<F>;
        MFQ_NVQ_CPU_CASE(1) MFQ_NVQ_CPU_CASE(2) MFQ_NVQ_CPU_CASE(3)
        MFQ_NVQ_CPU_CASE(5) MFQ_NVQ_CPU_CASE(7) MFQ_NVQ_CPU_CASE(8)
        MFQ_NVQ_CPU_CASE(9) MFQ_NVQ_CPU_CASE(10) MFQ_NVQ_CPU_CASE(11)
        MFQ_NVQ_CPU_CASE(12) MFQ_NVQ_CPU_CASE(13) MFQ_NVQ_CPU_CASE(14)
        MFQ_NVQ_CPU_CASE(15)
#undef MFQ_NVQ_CPU_CASE
        default: throw std::runtime_error("unsupported CPU NVQ format");
    }
}

CpuProjectionRows make_cpu_projection_rows(const NvqWeight& w) {
    MFQ_RUNTIME_CHECK(w.indices_packed.is_cpu() && w.gs==24 && w.sub_bits>0 && w.sub_bits<=4,
        "original-FP32 NVQ rows require canonical CPU storage");
    CpuProjectionRows p; p.width=static_cast<int>(w.neuron_len); p.outputs=static_cast<int>(w.out);
    const int format=static_cast<int>(w.kernel_format);
    const int vector=cpu_nvq_d4(format) ? 4 : 8;
    auto& v=p.nvq;
    v.indices=w.indices_packed.data_ptr<uint8_t>(); v.aux=w.aux_packed.data_ptr<uint8_t>();
    v.index_bytes=w.indices_packed.numel(); v.aux_bytes=w.aux_packed.numel();
    v.nvec=static_cast<int>((w.neuron_len+vector-1)/vector);v.nsign=static_cast<int>((w.neuron_len+7)/8);
    v.groups=static_cast<int>(w.ng);v.sign_mode=static_cast<int>(w.sign_mode);
    v.states=w.sub_scale_packed.data_ptr<uint8_t>();v.state_bytes=w.sub_scale_packed.numel();
    v.anchors=w.neuron_scale.data_ptr<float>();v.width=w.neuron_len;v.state_bits=static_cast<int>(w.sub_bits);
    const auto* metadata=w.codebook.data_ptr<int8_t>();
    for(uint32_t state=0;state<(1u<<w.sub_bits);++state) {
        v.multipliers[state]=cpu_nvq_scale(metadata,format,1.0f,state);
        v.banks[state]=format==7 ? metadata+64+state*32 : format==9 ? metadata+64+state*64*8 : cpu_nvq_codebook(metadata,format,state);
    }
    p.nvq_rows=w.dense_groups ? mfq::cpu::nvq_dense_rows_dot_kernel(format) :
        (format==1 || format==8) && v.aux_bytes==0?nullptr:mfq::cpu::nvq_rows_dot_kernel(format);
    MFQ_RUNTIME_CHECK(!w.dense_groups || p.nvq_rows,"compact NVQ CPU decoder unavailable");
    if(!p.nvq_rows) {
        const auto decoder=cpu_nvq_decoder(format);
        const auto dot=mfq::cpu::scaled_i8_dot_kernel();
        p.scalar_rows=[v,decoder,dot,format](const float* x,int batch,int stride,float* y,int out,int begin,int end) {
            for(int row=begin;row<end;++row) {
                for(int t=0;t<batch;++t) y[t*out+row]=0;
                for(int group=0;group<v.groups;++group) {
                    const auto state=cpu_nvq_state(v.indices,v.states,v.state_bytes,
                        int64_t(row)*v.groups+group,v.state_bits,format);
                    const float scale=v.anchors[row]*v.multipliers[state];
                    alignas(32) int8_t codes[24];decoder(v,row,group,state,codes);
                    const int valid=static_cast<int>(std::min<int64_t>(24,v.width-int64_t(group)*24));
                    for(int t=0;t<batch;++t) y[t*out+row]+=dot(codes,x+int64_t(t)*stride+group*24,valid,scale);
                }
            }
        };
    }
    return p;
}
static mfq_tensor_backend::Tensor nvq_matmul_cpu(
        const NvqWeight & w,
        mfq_tensor_backend::Tensor x) {
    MFQ_RUNTIME_CHECK(!x.is_cuda(), "CPU NVQ GEMV requires CPU activations");
    MFQ_RUNTIME_CHECK(
        !w.indices_packed.is_cuda() && !w.sub_scale_packed.is_cuda() &&
        !w.neuron_scale.is_cuda() && !w.codebook.is_cuda() && !w.aux_packed.is_cuda(),
        "CPU NVQ GEMV requires CPU-resident weights");
    MFQ_RUNTIME_CHECK(w.gs == 24,
        "CPU NVQ GEMV currently requires 24-value groups");
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat32), w.neuron_len).contiguous();
    const int64_t rows = x.size(0);
    const float* input = x.data_ptr<float>();
    const auto dot = mfq::cpu::scaled_i8_dot_kernel();
    const int64_t outputs = w.out;
    const uint8_t * scales = w.sub_scale_packed.data_ptr<uint8_t>();
    const float * anchors = w.neuron_scale.data_ptr<float>();
    const int8_t * metadata = w.codebook.data_ptr<int8_t>();
    const int64_t scale_bytes = w.sub_scale_packed.numel();
    const int format=static_cast<int>(w.kernel_format);
    const auto decoder=cpu_nvq_decoder(format);
    const auto fused_rows=w.dense_groups ? mfq::cpu::nvq_dense_rows_dot_kernel(format) :
        (format==1 || format==8) && w.aux_packed.numel()==0?nullptr:mfq::cpu::nvq_rows_dot_kernel(format);
    MFQ_RUNTIME_CHECK(!w.dense_groups || fused_rows,"compact NVQ CPU decoder unavailable");
    const int vector_size=cpu_nvq_d4(format) ? 4 : 8;
    MFQ_RUNTIME_CHECK(w.sub_bits>0 && w.sub_bits<=4,"CPU NVQ state width must be 1..4");
    CpuNvqDecodeView view{w.indices_packed.data_ptr<uint8_t>(),
        w.aux_packed.data_ptr<uint8_t>(), w.indices_packed.numel(),w.aux_packed.numel(),
        static_cast<int>((w.neuron_len+vector_size-1)/vector_size),
        static_cast<int>((w.neuron_len+7)/8),static_cast<int>(w.ng),static_cast<int>(w.sign_mode)};
    for (uint32_t state=0; state<(1u<<w.sub_bits); ++state) {
        view.multipliers[state]=cpu_nvq_scale(metadata,format,1.0f,state);
        view.banks[state]=format==7 ? metadata+64+state*32 :
            format==9 ? metadata+64+state*64*8 : cpu_nvq_codebook(metadata,format,state);
    }
    if (fused_rows) {
        view.states=scales; view.state_bytes=scale_bytes;
        view.anchors=anchors; view.width=w.neuron_len;
        view.state_bits=static_cast<int>(w.sub_bits);
        auto result=mfq_tensor_backend::empty({rows,outputs},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32));
        auto* output=result.data_ptr<float>();
        mfq_parallel_for(0,outputs,1,[&](int64_t begin,int64_t end) {
            fused_rows(view,input,rows,w.neuron_len,output,outputs,begin,end);
        });
        return result.to(mfq_tensor_backend::kFloat16);
    }
    auto result=mfq_tensor_backend::empty({rows,outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    auto* output=result.data_ptr<mfq_half>();
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            for (int group = 0; group < w.ng; ++group) {
                const int64_t scale_linear = neuron * w.ng + group;
                const uint32_t state = cpu_nvq_state(view.indices,scales,scale_bytes,
                    scale_linear,static_cast<int>(w.sub_bits),format);
                const float scale = anchors[neuron]*view.multipliers[state];
                const int valid = static_cast<int>(std::min<int64_t>(
                    w.gs, w.neuron_len - static_cast<int64_t>(group) * w.gs));
                alignas(32) int8_t weights[24];
                decoder(view,neuron,group,state,weights);
                for (int64_t row = 0; row < rows; ++row) {
                    accumulators[static_cast<size_t>(row)] += dot(weights,
                        input + row*w.neuron_len + static_cast<int64_t>(group)*w.gs,
                        valid, scale);
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] =
                    mfq_half(accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}
mfq_tensor_backend::Tensor nvq_matmul(
        CudaProfiler& profiler,
        const NvqWeight& w,
        mfq_tensor_backend::Tensor x) {
    if (!x.is_cuda()) return nvq_matmul_cpu(w, std::move(x));
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    int M = (int)x.size(0);
    const NvqMatmulPath path = select_nvq_matmul_path(w, M);
    if (path == NvqMatmulPath::Gemv) {
        NvqWorkspace & ws = w.workspace(M);
        return profiler.measure("nvq.gemv", [&]() {
            return nvq_gemv_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::Mmq) {
        NvqWorkspace & ws = w.workspace(M);
        return profiler.measure("nvq.mma24", [&]() {
            return nvq_mmq_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::OnlineF16) {
        return profiler.measure("nvq.gemm_online_f16", [&]() {
            return nvq_gemm_f16_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode);
        });
    }
    auto weight = profiler.measure("nvq.dequant", [&]() { return nvq_dequant(w); });
    return profiler.measure("nvq.gemm", [&]() {
        return nint_cublas_gemm_nt_f32acc_cuda(x, weight);
    });
}

mfq_tensor_backend::Tensor nvq_matmul_input_mul(
    CudaProfiler& profiler,
    const NvqWeight & w,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor gate,
    int mode) {
    MFQ_RUNTIME_CHECK(mode == 1 || mode == 2, "NVQ input gate mode must be 1(sigmoid) or 2(silu)");
    if (!x.is_cuda()) {
        x = x.contiguous().to(mfq_tensor_backend::kFloat16);
        gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
        MFQ_RUNTIME_CHECK(x.sizes() == gate.sizes(), "NVQ x and gate shapes must match");
        auto value = mode == 1
            ? x * mfq_tensor_backend::sigmoid(gate)
            : x * mfq_tensor_backend::silu(gate);
        return nvq_matmul_cpu(w, value);
    }
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(x.sizes() == gate.sizes(), "NVQ x and gate shapes must match");
    x = pad_last(x, w.neuron_len);
    gate = pad_last(gate, w.neuron_len);
    int M = (int)x.size(0);
    const NvqMatmulPath path = select_nvq_matmul_path(w, M);
    if (path == NvqMatmulPath::Gemv) {
        NvqWorkspace & ws = w.workspace(M);
        return profiler.measure("nvq.gemv_gate", [&]() {
            return nvq_gemv_gate_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, gate, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::Mmq) {
        NvqWorkspace & ws = w.workspace(M);
        return profiler.measure("nvq.mma24_gate", [&]() {
            return nvq_mmq_gate_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, gate, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, mode, ws.qx, ws.xscale);
        });
    }
    mfq_tensor_backend::Tensor value = mode == 1 ? x * mfq_tensor_backend::sigmoid(gate) : x * mfq_tensor_backend::silu(gate);
    return nvq_matmul(profiler, w, value);
}

bool nvq_pair_compatible(const NvqWeight & first, const NvqWeight & second) {
    return first.format == second.format && first.kernel_format == second.kernel_format &&
           first.gs == second.gs &&
           first.neuron_len == second.neuron_len && first.ng == second.ng;
}

bool nvq_fusion_enabled(const CudaExecutionConfig& config) {
    return config.nvq_fusion;
}

mfq_tensor_backend::Tensor nvq_matmul_multi2(
    CudaProfiler& profiler,
    const NvqWeight & first,
    const NvqWeight & second,
    mfq_tensor_backend::Tensor x) {
    if (!nvq_pair_compatible(first, second)) {
        throw std::runtime_error("NVQ multi-projection requires compatible formats and input layouts");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), first.neuron_len);
    const int M = (int)x.size(0);
    if (M > 8) {
        return mfq_tensor_backend::cat(
            {nvq_matmul(profiler, first, x),
             nvq_matmul(profiler, second, x)}, -1);
    }
    NvqWorkspace & ws = first.workspace(M);
    return profiler.measure("nvq.gemv_multi2", [&]() {
        return nvq_gemv_multi2_ws_cuda(
            first.indices_packed, first.aux_packed, first.sub_scale_packed,
            first.neuron_scale, first.codebook,
            second.indices_packed, second.aux_packed, second.sub_scale_packed,
            second.neuron_scale, second.codebook,
            x, first.neuron_len, first.gs,
            first.sub_bits, first.kernel_format, first.sign_mode,
            second.sub_bits, second.kernel_format, second.sign_mode,
            ws.qx, ws.xscale);
    });
}

mfq_tensor_backend::Tensor nvq_matmul_swiglu(
    CudaProfiler& profiler,
    const NvqWeight & gate,
    const NvqWeight & up,
    mfq_tensor_backend::Tensor x) {
    if (!nvq_pair_compatible(gate, up) || gate.out != up.out) {
        throw std::runtime_error("NVQ SwiGLU requires compatible equal-width gate/up weights");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), gate.neuron_len);
    if (x.size(0) != 1) {
        auto pair = nvq_matmul_multi2(profiler, gate, up, x);
        auto parts = pair.split_with_sizes({gate.out, up.out}, -1);
        return mfq_tensor_backend::silu(parts[0]) * parts[1];
    }
    NvqWorkspace & ws = gate.workspace(1);
    return profiler.measure("nvq.gemv_swiglu", [&]() {
        return nvq_gemv_swiglu_ws_cuda(
            gate.indices_packed, gate.aux_packed, gate.sub_scale_packed,
            gate.neuron_scale, gate.codebook,
            up.indices_packed, up.aux_packed, up.sub_scale_packed,
            up.neuron_scale, up.codebook,
            x, gate.neuron_len, gate.gs,
            gate.sub_bits, gate.kernel_format, gate.sign_mode,
            up.sub_bits, up.kernel_format, up.sign_mode,
            ws.qx, ws.xscale);
    });
}

mfq_tensor_backend::Tensor nvq_ffn_swiglu_down(
    CudaProfiler& profiler,
    const NvqWeight & gate,
    const NvqWeight & up,
    const NvqWeight & down,
    mfq_tensor_backend::Tensor x,
    MfqOptional<mfq_tensor_backend::Tensor> residual) {
    if (!nvq_pair_compatible(gate, up) || gate.out != up.out || gate.out != down.neuron_len) {
        throw std::runtime_error("NVQ fused FFN weight layouts are incompatible");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), gate.neuron_len);
    if (x.size(0) != 1 || (down.gs != 24 && down.gs != 28 && down.gs != 32)) {
        auto output = nvq_matmul(
            profiler, down,
            nvq_matmul_swiglu(profiler, gate, up, x));
        return residual.has_value()
            ? acc_cuda(residual.value(), output)
            : output;
    }
    NvqWorkspace & input_ws = gate.workspace(1);
    NvqWorkspace & output_ws = down.workspace(1);
    if (!input_ws.swiglu_scratch.defined() || input_ws.swiglu_scratch.numel() < gate.out) {
        input_ws.swiglu_scratch = mfq_tensor_backend::empty(
            {gate.out}, gate.indices_packed.options().dtype(
                mfq_tensor_backend::kFloat32));
    }
    profiler.measure("nvq.ffn_swiglu_quant", [&]() {
        nvq_ffn_swiglu_quant_ws_cuda(
            gate.indices_packed, gate.aux_packed, gate.sub_scale_packed,
            gate.neuron_scale, gate.codebook,
            up.indices_packed, up.aux_packed, up.sub_scale_packed,
            up.neuron_scale, up.codebook,
            x, gate.neuron_len, gate.gs,
            gate.sub_bits, gate.kernel_format, gate.sign_mode,
            up.sub_bits, up.kernel_format, up.sign_mode, down.gs,
            input_ws.qx, input_ws.xscale,
            output_ws.qx, output_ws.xscale, input_ws.swiglu_scratch);
        return 0;
    });
    return profiler.measure(
        residual.has_value() ? "nvq.gemv_qx_residual" : "nvq.gemv_qx",
        [&]() {
        if (residual.has_value()) {
            return nvq_gemv_qx_residual_ws_cuda(
                down.indices_packed, down.aux_packed, down.sub_scale_packed,
                down.neuron_scale, down.codebook,
                down.neuron_len, down.gs, down.sub_bits,
                down.kernel_format, down.sign_mode,
                output_ws.qx, output_ws.xscale, residual.value());
        }
        return nvq_gemv_qx_ws_cuda(
            down.indices_packed, down.aux_packed, down.sub_scale_packed,
            down.neuron_scale, down.codebook,
            down.neuron_len, down.gs, down.sub_bits, down.kernel_format, down.sign_mode,
            output_ws.qx, output_ws.xscale);
    });
}

mfq_tensor_backend::Tensor nvq_embedding(const NvqWeight & w, mfq_tensor_backend::Tensor token_ids) {
    return nvq_embedding_lookup_cuda(
        w.indices_packed, w.aux_packed, w.sub_scale_packed,
        w.neuron_scale, w.codebook, token_ids, w.neuron_len, w.gs,
        w.sub_bits, w.kernel_format, w.sign_mode);
}
mfq_tensor_backend::Tensor nvq_dequant(const NvqWeight & w) {
    return nvq_dequant_cuda(
        w.indices_packed, w.aux_packed, w.sub_scale_packed,
        w.neuron_scale, w.codebook, w.neuron_len, w.gs,
        w.sub_bits, w.kernel_format, w.sign_mode);
}
bool nvq_fused_residual_format(std::int64_t kernel_format) {
    return kernel_format == kNvq2JscXlGroupExecKernelFormat ||
        kernel_format == kNvq3JscLGroupExecKernelFormat;
}
