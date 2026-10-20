#include "mlx_csa2.h"
#include "runtime/mlx_kernel_prepare.h"
#include "runtime/mlx_platform.h"
#include "mfq_mfe_prefill_embedded.h"
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/primitives.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace mfq::metal {
namespace {
using namespace mlx::core;
using Kernel = fast::CustomKernelFunction;
using Templates = std::vector<std::pair<std::string, fast::TemplateArg>>;
#include "kernels/mfq_csa2_kernels.inc"
#include "kernels/mfq_csa2_nax.inc"

constexpr char kPackHeader[] = R"CSA2(
inline uchar csa2_fp8_bits(float f) {
 uint bits=as_type<uint>(f), sign=bits&0x80000000u; bits^=sign;
 uchar out;
 if(bits>=(543u<<21))out=0x7e;
 else if(bits<(121u<<23)){
  const uint denorm=141u<<23;
  out=uchar(as_type<uint>(as_type<float>(bits)+as_type<float>(denorm))-denorm);
 }else{const uint odd=(bits>>20)&1;bits+=((uint)(7-127)<<23)+0x7ffff+odd;out=uchar(bits>>20);}
 return out|uchar(sign>>24);
}
inline float csa2_fp8_round(float x) {
 const float a=min(abs(x),448.0f);
 const int exp=max(int(floor(log2(max(a,0x1p-9f))))-3,-9);
 const float step=as_type<float>(uint(exp+127)<<23);
 return sign(x)*min(rint(a/step)*step,448.0f);
}
)CSA2";
constexpr char kPack[] = R"CSA2(
 const uint i=thread_position_in_grid.x;
 uint n=1;for(int axis=0;axis<x_ndim;++axis)n*=x_shape[axis];
 const float v=i<n?float(x[i]):0.0f;
 float amax=abs(v);
 for(uint m=1;m<GS;m<<=1)amax=max(amax,simd_shuffle_xor(amax,m));
 const float limit=BITS==8?448.0f:6.0f;
 amax=max(amax,limit*(E4?0x1p-9f:0x1p-126f));
 const int exponent=max(int(ceil(log2(amax/limit))),-126);
 const float scale=E4?csa2_fp8_round(amax/limit):as_type<float>(uint(exponent+127)<<23);
 const float scaled=clamp(v/scale,-limit,limit), a=abs(scaled);
 uint code=0;
 if(BITS==8)code=csa2_fp8_bits(csa2_fp8_round(scaled));
 else {
  code=(a>0.25f)+(a>=0.75f)+(a>1.25f)+(a>=1.75f)+(a>2.5f)+(a>=3.5f)+(a>5.0f);
  code|=scaled<0?8:0;
 }
 const uint partner=simd_shuffle_xor(code,1);
 if(i>=n)return;
 const uint row=i/D,d=i%D, base=row*(D*BITS/8+D/GS);
 if(BITS==8)packed[base+d]=uchar(code);
 else if((d&1)==0)packed[base+d/2]=uchar(code|(partner<<4));
 if(d%GS==0)packed[base+D*BITS/8+d/GS]=E4?csa2_fp8_bits(scale):uchar(exponent+127);
)CSA2";
constexpr char kUnpack[] = R"CSA2(
 const uint i=thread_position_in_grid.x;
 uint n=D;for(int axis=0;axis<packed_ndim-1;++axis)n*=packed_shape[axis];
 if(i>=n)return;
 const uint row=i/D,d=i%D,base=row*(D*BITS/8+D/GS);
 const uchar s=packed[base+D*BITS/8+d/GS];
 const float scale=E4?v41_fp8(s):as_type<float>(uint(s)<<23);
 const float v=BITS==8?v41_fp8(packed[base+d]):v41_fp4((packed[base+d/2]>>((d%2)*4))&15);
 out[i]=T(v*scale);
)CSA2";

enum Kind { attention, attention_mma, merge, index, index_mma, topk, topk_merge, radix, pack, unpack };
const Kernel& kernel(Kind kind) {
    static const std::array<Kernel, 10> kernels = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        const auto make = [&](const char* name, std::vector<std::string> inputs,
            std::vector<std::string> outputs, const char* source, const std::string& header) {
            return fast::metal_kernel(name, inputs, outputs, source, header, true, false, options);
        };
        return std::array<Kernel, 10>{
            make("mfq_csa2_attention", {"q","window","pooled","wi","ci","meta","scalep"}, {"partial"}, kCsa2Attention, kCsa2Header),
            make("mfq_csa2_attention_mma", {"q","window","pooled","wi","ci","meta","scalep"}, {"partial"}, kCsa2AttentionMma, kCsa2Header),
            make("mfq_csa2_merge", {"partial","sink","meta"}, {"out"}, kCsa2Merge, kCsa2Header),
            make("mfq_csa2_index", {"q","keys","weights","candidates","meta"}, {"scores"}, kCsa2Index, kCsa2Header),
            make("mfq_csa2_index_mma", {"q","keys","weights","candidates","meta"}, {"scores"}, kCsa2IndexMma, kCsa2Header),
            make("mfq_csa2_topk", {"scores","ids","meta"}, {"selected_scores","selected_ids"}, kCsa2Topk, kCsa2Header),
            make("mfq_csa2_topk_merge", {"a_scores","a_ids","b_scores","b_ids","meta"}, {"selected_scores","selected_ids"}, kCsa2TopkMerge, kCsa2Header),
            make("mfq_csa2_radix_topk", {"scores","meta"}, {"ids"}, kCsa2RadixTopk, kCsa2RadixHeader),
            make("mfq_csa2_pack", {"x"}, {"packed"}, kPack, std::string(kCsa2Header)+kPackHeader),
            make("mfq_csa2_unpack", {"packed"}, {"out"}, kUnpack, kCsa2Header),
        };
    }();
    return kernels.at(kind);
}

array slice_axis(const array& x, int axis, int begin, int end) {
    Shape a(x.ndim(), 0), b=x.shape(); a[axis]=begin; b[axis]=end;
    return slice(x,a,b);
}
array metadata(std::initializer_list<int> values) {
    return array(values.begin(), Shape{int(values.size())}, int32);
}


#include "mlx_csa2_short.inc"

struct PackedParams {
    int B,H,qL,localL,pooledL,topk,local_window,compress_ratio,q_offset;
    float scale;
    std::int64_t Q[3],Local[3],Pooled[2],Topk[3],O[3];
};

class Csa2Prefill final : public Primitive, public MlxPreparableKernel {
public:
    Csa2Prefill(Stream stream, int offset, int ratio, int window, Dtype dtype, bool unpack)
        : Primitive(stream), offset_(offset), ratio_(ratio), window_(window), dtype_(dtype), unpack_(unpack) {}
    std::string name_string() const {
        return std::string("mfq_csa2_prefill_")+(dtype_==float16?"f16":"bf16")+"_directq"+(unpack_?"_unpacked":"");
    }
    std::string preparation_key() const override { return name_string(); }
    MTL::Library* library() {
        CompileOptions options; options.math_mode=MathMode::Fast;
        return kernel_device(stream().device).get_library("mfq_csa2_prefill_v1", options, [] {
            return std::string(detail::kSteelAttentionSource)+detail::kCsa2PrefillSource;
        });
    }
    void prepare_gpu() override {
        auto& d=kernel_device(stream().device); auto* lib=library();
        (void)d.get_kernel(name_string(),lib);
        if(unpack_)(void)d.get_kernel(dtype_==float16?"mfq_csa2_unpack_f16":"mfq_csa2_unpack_bf16",lib);
    }
    void eval_cpu(const std::vector<array>&,std::vector<array>&) override { throw std::runtime_error("CSA2 requires Metal"); }
    void eval_gpu(const std::vector<array>& in,std::vector<array>& out) override {
        const auto& q=in[0];const auto& local=in[1];const auto& pooled=in[2];const auto& ids=in[3];
        auto& result=out[0]; result.set_data(allocator::malloc(result.nbytes()));
        PackedParams p{q.shape(0),q.shape(1),q.shape(2),local.shape(1),pooled.shape(1),ids.shape(2),window_,ratio_,offset_,1.0f/std::sqrt(512.0f),
            {q.strides(0),q.strides(1),q.strides(2)}, {local.strides(0),0,local.strides(1)},
            {pooled.strides(0),pooled.strides(1)},{ids.strides(0),0,ids.strides(1)},
            {result.strides(0),result.strides(1),result.strides(2)}};
        auto& device=kernel_device(stream().device);auto* lib=library();
        auto& encoder=mlx::core::metal::get_command_encoder(stream());
        auto local_values=local, pooled_values=pooled;
        if(unpack_) {
            local_values=array(Shape{p.B,p.localL,512},dtype_,nullptr,{});
            pooled_values=array(Shape{p.B,std::max(p.pooledL,1),512},dtype_,nullptr,{});
            local_values.set_data(allocator::malloc(local_values.nbytes()));
            pooled_values.set_data(allocator::malloc(pooled_values.nbytes()));
            encoder.add_temporary(local_values);encoder.add_temporary(pooled_values);
            encoder.set_compute_pipeline_state(device.get_kernel(dtype_==float16?"mfq_csa2_unpack_f16":"mfq_csa2_unpack_bf16",lib));
            encoder.set_input_array(local,0);encoder.set_input_array(pooled,1);
            encoder.set_output_array(local_values,2);encoder.set_output_array(pooled_values,3);encoder.set_bytes(p,4);
            encoder.dispatch_threads(MTL::Size(std::size_t(p.B)*(p.localL+p.pooledL)*512,1,1),MTL::Size(256,1,1));
            p.Local[0]=std::int64_t(p.localL)*512*2;p.Local[2]=512*2;
            p.Pooled[0]=std::int64_t(p.pooledL)*512*2;p.Pooled[1]=512*2;
        }
        encoder.set_compute_pipeline_state(device.get_kernel(name_string(),lib));
        encoder.set_input_array(q,0);encoder.set_input_array(local_values,1);encoder.set_input_array(pooled_values,2);
        encoder.set_input_array(ids,3);encoder.set_input_array(in[4],4);encoder.set_output_array(result,5);encoder.set_bytes(p,6);
        encoder.dispatch_threadgroups(MTL::Size(p.qL,p.B,1),MTL::Size(32,8,1));
    }
    const char* name() const override { return "Csa2Prefill"; }
    bool is_equivalent(const Primitive& other) const override {
        const auto& x=static_cast<const Csa2Prefill&>(other);
        return offset_==x.offset_&&ratio_==x.ratio_&&window_==x.window_&&dtype_==x.dtype_&&unpack_==x.unpack_;
    }
private:
    int offset_,ratio_,window_;Dtype dtype_;bool unpack_;
};

}

array mlx_csa2_pack(const array& value,int bits,int group_size,bool e4m3_scale) {
    if(value.ndim()<2 || value.shape(-1)<=0 || value.shape(-1)%group_size ||
        (bits!=4&&bits!=8)||(group_size!=16&&group_size!=32)|| (e4m3_scale&&bits!=4))
        throw std::invalid_argument("invalid CSA2 packed activation geometry");
    auto input=contiguous(value); auto shape=input.shape();
    const int width=shape.back();shape.back()=width*bits/8+width/group_size;
    return kernel(pack)({input},{shape},{uint8},{int((input.size()+255)/256*256),1,1},{256,1,1},
        {{"D",width},{"BITS",bits},{"GS",group_size},{"E4",e4m3_scale}},std::nullopt,false,{}).front();
}

array mlx_csa2_unpack(const array& value,int bits,int group_size,bool e4m3_scale,Dtype dtype) {
    if(value.ndim()<2||value.dtype()!=uint8||(bits!=4&&bits!=8)||(group_size!=16&&group_size!=32))
        throw std::invalid_argument("invalid CSA2 packed activation");
    const int stride=group_size*bits/8+1;
    if(value.shape(-1)%stride)throw std::invalid_argument("invalid CSA2 packed row width");
    auto shape=value.shape();const int width=shape.back()/stride*group_size;shape.back()=width;
    const auto count=value.size()/value.shape(-1)*width;
    if (!count) return zeros(shape,dtype);
    return kernel(unpack)({contiguous(value)},{shape},{dtype},{int((count+255)/256*256),1,1},{256,1,1},
        {{"D",width},{"BITS",bits},{"GS",group_size},{"E4",e4m3_scale},{"T",dtype}},std::nullopt,false,{}).front();
}

array mlx_csa2_index_scores(const array& query,const array& keys,const array& weights,
    int position,int ratio,const std::optional<array>& candidates) {
    if(query.ndim()!=4||keys.ndim()!=3||query.shape(0)!=keys.shape(0)||query.shape(3)%32||
        keys.dtype()!=uint8||weights.shape()!=Shape{query.shape(0),query.shape(1),query.shape(2)}||ratio<=0)
        throw std::invalid_argument("invalid packed CSA2 indexer geometry");
    const int batch=query.shape(0),length=query.shape(1),heads=query.shape(2),dim=query.shape(3);
    const int width=candidates?candidates->shape(-1):keys.shape(1);
    if(width==0)return zeros({batch,length,0},float32);
    if(batch>1){std::vector<array> values;for(int b=0;b<batch;++b)
        values.push_back(mlx_csa2_index_scores(slice_axis(query,0,b,b+1),slice_axis(keys,0,b,b+1),
            slice_axis(weights,0,b,b+1),position,ratio,candidates?std::optional<array>{slice_axis(*candidates,0,b,b+1)}:std::nullopt));
        return concatenate(values,0);}
    const bool mma=dim<=128&&heads>=8, split=mma&&heads>=32;
    const int cols=mma?(split?16:32):4;
    auto result=kernel(mma?index_mma:index)({contiguous(query),contiguous(keys),contiguous(astype(weights,float32)/std::sqrt(float(dim*heads))),
        candidates?contiguous(astype(*candidates,int32)):zeros({1},int32),metadata({heads,keys.shape(1),width,position,ratio,0})},
        {{1,length,width}},{float32},{(width+cols-1)/cols*128,length,1},{128,1,1},
        {{"D",dim},{"CANDIDATES",candidates.has_value()},{"HEAD_SPLIT",split},{"HEADS",heads},{"RATIO",ratio}},std::nullopt,false,{}).front();
    return result;
}

array mlx_csa2_topk(const array& scores,int count,const std::optional<array>& ids) {
    if(scores.ndim()!=3||scores.dtype()!=float32||count<0||(ids&&ids->shape()!=scores.shape()))
        throw std::invalid_argument("invalid CSA2 top-k input");
    const int batch=scores.shape(0),length=scores.shape(1),width=scores.shape(2);
    count=std::min(count,width);
    if(!count)return zeros({batch,length,0},int32);
    if(batch>1){std::vector<array> results;for(int b=0;b<batch;++b)
        results.push_back(mlx_csa2_topk(slice_axis(scores,0,b,b+1),count,ids?std::optional<array>{slice_axis(*ids,0,b,b+1)}:std::nullopt));
        return concatenate(results,0);}
    array selected(0);
    if(!ids&&length>8&&width>=4096){
        selected=kernel(radix)({contiguous(scores),array({std::uint32_t(count)})},{{1,length,count}},{int32},
            {256*length,1,1},{256,1,1},{},std::nullopt,false,{}).front();
    }else{
        int tile=32;const int limit=length==1?4096:std::max(1024,2*count);
        while(tile<std::min(width,limit))tile*=2;
        const int threads=std::min(256,tile);int tiles=(width+tile-1)/tile;
        auto winners=kernel(topk)({contiguous(scores),ids?contiguous(astype(*ids,int32)):zeros({1},int32),
            metadata({width,0,width,0,1,count,tiles})},{{1,length,tiles*count},{1,length,tiles*count}},{float32,int32},
            {tiles*threads,length,1},{threads,1,1},{{"TILE",tile},{"EXPLICIT",ids.has_value()},{"BLOCK",1},{"FORCE",false},{"THREADS",threads},{"RATIO",1}},std::nullopt,false,{});
        while(tiles>1){
            const int fanin=count<=512?4:2,groups=(tiles+fanin-1)/fanin;
            winners=kernel(topk_merge)({winners[0],winners[1],winners[0],winners[1],metadata({winners[0].shape(-1),winners[0].shape(-1),tiles,count})},
                {{1,length,groups*count},{1,length,groups*count}},{float32,int32},{((fanin*count+255)/256)*256,length,groups},{256,1,1},
                {{"GROUPED",true},{"FANIN",fanin}},std::nullopt,false,{});
            tiles=groups;
        }
        selected=winners[1];
        return sort(where(winners[0]>array(-std::numeric_limits<float>::infinity()),selected,array(-1,int32)),-1);
    }
    auto valid=take_along_axis(scores,selected,-1)>array(-std::numeric_limits<float>::infinity());
    return sort(where(valid,selected,array(-1,int32)),-1);
}

array mlx_csa2_attention(const array& query,const array& local,const std::optional<array>& pooled,
    int pool_length,const array& topk_ids,const array& sinks,int position,int ratio,int window) {
    if(query.ndim()!=4||local.ndim()!=3||query.shape(0)!=local.shape(0)||query.shape(3)%32||local.dtype()!=uint8||
        ratio<=0||window<=0||pool_length<0||(!pooled&&pool_length))throw std::invalid_argument("invalid packed CSA2 attention geometry");
    const int batch=query.shape(0),length=query.shape(1),heads=query.shape(2),dim=query.shape(3);
    auto pool=pooled?slice_axis(*pooled,1,0,pool_length):zeros({batch,0,dim/2+dim/16},uint8);
    if(batch==1&&length>8&&heads==64&&dim==512&&query.dtype()==float16&&
        std::size_t(local.shape(1)+pool_length)*dim*2<=80*1024*1024&&csa2_nax_supported(to_stream({})))
        return array(query.shape(), query.dtype(),
            std::make_shared<Csa2Short>(to_stream({}), dim, position, window, true),
            {contiguous(query), view(mlx_csa2_unpack(local,8,32,false,float16),uint8),
                view(mlx_csa2_unpack(pool,4,16,true,float16),uint8),
                contiguous(astype(topk_ids, int32)), contiguous(astype(sinks, float32))});
    if(length>8&&heads==64&&dim==512&&(query.dtype()==float16||query.dtype()==bfloat16)){
        auto q=transpose(query,{0,2,1,3});
        const bool unpack_rows=length>=256&&std::size_t(batch)*(local.shape(1)+std::max(pool_length,1))*dim*2<=80*1024*1024;
        auto result=array(q.shape(),q.dtype(),std::make_shared<Csa2Prefill>(to_stream({}),position,ratio,window,q.dtype(),unpack_rows),
            {q,contiguous(local),contiguous(pool),contiguous(astype(topk_ids,uint32)),contiguous(astype(sinks,float32))});
        return transpose(result,{0,2,1,3});
    }
    if(batch>1){std::vector<array> results;for(int b=0;b<batch;++b)
        results.push_back(mlx_csa2_attention(slice_axis(query,0,b,b+1),slice_axis(local,0,b,b+1),
            slice_axis(pool,0,b,b+1),pool_length,slice_axis(topk_ids,0,b,b+1),sinks,position,ratio,window));
        return concatenate(results,0);}
    if (length <= 8 && dim <= 512 && query.dtype() == float16)
        return array(query.shape(), query.dtype(),
            std::make_shared<Csa2Short>(to_stream({}), dim, position, window),
            {contiguous(query), contiguous(local), contiguous(pool),
                contiguous(astype(topk_ids, int32)), contiguous(astype(sinks, float32))});
    const int slots=position==0?std::min(window,length):window;
    auto positions=reshape(arange(position,position+length,int32),{1,length,1});
    auto absolute=positions-window+1+reshape(arange(slots,int32),{1,1,slots});
    if(position==0)absolute=maximum(positions-window+1,array(0,int32))+reshape(arange(slots,int32),{1,1,slots});
    auto valid=(absolute>=array(std::max(0,position-(local.shape(1)-length)),int32))&&(absolute<=positions);
    auto local_ids=length==1?remainder(absolute,array(window,int32)):absolute-array(position-(local.shape(1)-length),int32);
    local_ids=where(valid,local_ids,array(-1,int32));
    const int compressed=topk_ids.shape(-1),chunk=length<=8?32:128,splits=std::max(1,(slots+compressed+chunk-1)/chunk);
    const bool mma=dim<=512&&heads>=8&&(length>8||slots+compressed>=512);const int group=mma?8:4;
    auto partial=kernel(mma?attention_mma:attention)({contiguous(query),contiguous(local),contiguous(pool),contiguous(local_ids),contiguous(topk_ids),
        metadata({heads,slots,compressed,local.shape(1),pool_length,splits}),array({1.0f/std::sqrt(float(dim))})},
        {{1,length,heads,splits,dim+2}},{float32},{(heads+group-1)/group*128,length,splits},{128,1,1},
        {{"D",dim},{"CHUNK",chunk}},std::nullopt,false,{}).front();
    return kernel(merge)({partial,contiguous(astype(sinks,float32)),metadata({heads,splits})},{query.shape()},{query.dtype()},
        {(heads+3)/4*128,length,1},{128,1,1},{{"D",dim},{"T",query.dtype()}},std::nullopt,false,{}).front();
}

}
