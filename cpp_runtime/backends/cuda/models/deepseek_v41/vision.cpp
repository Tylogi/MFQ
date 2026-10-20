#include "vision.h"
#include "core/grid_vision_component.h"
#include "mfq_paged_prefix_cache.h"

namespace mfq::cuda::deepseek_v41_runtime {
namespace {
namespace memory=mfq::cuda::grid_vision_runtime::detail;

struct HostLoading {
    bool& value;bool previous;
    explicit HostLoading(CudaExecutionContext& execution):value(execution.loading_cpu_layer),previous(value){value=true;}
    ~HostLoading(){value=previous;}
};

Tensor vector_weight(CudaExecutionContext& execution,const mfq::ModelSource& source,
                     const std::string& name,int64_t width) {
    auto value=load_dense_gpu(execution,source,name);
    MFQ_RUNTIME_CHECK(value.dim()==1 && value.size(0)==width,"DeepSeek-V4.1 vision vector shape mismatch: "+name);
    return value;
}

struct Affine {
    QuantLinear linear;
    Tensor bias;
    static Affine load(CudaExecutionContext& execution,const mfq::ModelSource& source,
                       const std::string& prefix,int64_t input,int64_t output) {
        Affine result{load_quant_linear(execution,source,prefix+".weight"),{}};
        MFQ_RUNTIME_CHECK(result.linear.neuron_len()==input && result.linear.out()==output,
                          "DeepSeek-V4.1 vision linear shape mismatch: "+prefix);
        if(has_tensor(source,prefix+".bias"))result.bias=vector_weight(execution,source,prefix+".bias",output);
        return result;
    }
    template<class F> void visit_tensors(F visit){memory::visit_linear(linear,visit);if(bias.defined())visit(bias);}
    Tensor operator()(CudaExecutionContext& execution,Tensor value) const {
        auto result=linear.forward(execution,std::move(value));
        return bias.defined()?result+bias.to(result.scalar_type()):result;
    }
};

struct BlockWeights {
    Tensor norm1,norm2;
    Affine qkv,output,gate_up,down;
    template<class F> void visit_tensors(F visit) {
        visit(norm1);visit(norm2);qkv.visit_tensors(visit);output.visit_tensors(visit);
        gate_up.visit_tensors(visit);down.visit_tensors(visit);
    }
};

struct AlignerWeights {
    Tensor norm;
    Affine input,output;
    template<class F> void visit_tensors(F visit){visit(norm);input.visit_tensors(visit);output.visit_tensors(visit);}
};

Tensor norm(Tensor value,const Tensor& weight) {
    return qwen_rms_norm(value.to(mfq_tensor_backend::kFloat32),weight.to(mfq_tensor_backend::kFloat32),1e-6,false)
        .to(value.scalar_type()).contiguous();
}

template<class Weights,class Forward> Tensor use_weights(CudaExecutionContext& execution,
        Weights weights,std::size_t activations,Forward forward) {
    // One vision block at a time: text-only loading owns no vision allocation,
    // and even a large tower needs no permanent RAM or VRAM weight copy.
    return memory::vision_workspace(execution,memory::weight_bytes(weights)+activations,[&] {
        auto device=memory::to_cuda(weights);
        return forward(device);
    });
}

struct Span {int64_t begin,end,height,width,rows,columns;};
std::vector<Span> validate_media(const CommonConfig& config,const std::vector<int64_t>& tokens,
                                 const MfqMultimodalInput& media) {
    const auto count=media.image_bounds.size()/4;
    const auto patch_width=3*config.vision.patch_size*config.vision.patch_size;
    if(media.processor!=MfqMultimodalProcessor::deepseek_v41 || media.processor_name!="deepseek_v41" ||
            !count || media.image_bounds.size()%4 || media.vision_grid.size()!=count*4 ||
            media.pixel_shape.size()!=3 || media.pixel_shape[0]!=int64_t(count) ||
            media.pixel_shape[1]<=0 || media.pixel_shape[2]!=patch_width ||
            media.vision_grid_shape!=std::vector<int64_t>{int64_t(count),4} ||
            !media.video_grid.empty() || tokens.empty())
        throw std::invalid_argument("DeepSeek-V4.1 vision payload geometry is invalid");
    const auto maximum=uint64_t(media.pixel_shape[1]);
    if(maximum>SIZE_MAX/uint64_t(patch_width)/count ||
            media.pixel_values.size()!=count*maximum*uint64_t(patch_width))
        throw std::invalid_argument("DeepSeek-V4.1 vision patch storage is invalid");
    for(const auto token:tokens)if(token<0 || token>=config.vocab)
        throw std::invalid_argument("DeepSeek-V4.1 vision prompt token is outside the vocabulary");
    std::vector<Span> spans;spans.reserve(count);int64_t previous=0;
    const auto ratio=config.vision.downsample_ratio;
    for(std::size_t i=0;i<count;++i) {
        const auto* bound=media.image_bounds.data()+4*i;
        const auto* grid=media.vision_grid.data()+4*i;
        const int64_t h=grid[0],w=grid[1],rows=grid[2],columns=grid[3];
        if(bound[0]!=0 || bound[1]!=int64_t(i) || h<=0 || w<=0 || h>int64_t(maximum)/w ||
                rows!=(h+ratio-1)/ratio || columns!=(w+ratio-1)/ratio ||
                bound[2]<previous || bound[3]>int64_t(tokens.size()) ||
                bound[3]-bound[2]!=rows*(columns+1)+2 || rows*columns>config.vision.max_image_tokens)
            throw std::invalid_argument("DeepSeek-V4.1 image span is invalid");
        for(auto position=bound[2];position<bound[3];++position)if(tokens[position]!=config.image_token_id)
            throw std::invalid_argument("DeepSeek-V4.1 image span contains a text token");
        spans.push_back({bound[2],bound[3],h,w,rows,columns});previous=bound[3];
    }
    return spans;
}
} // namespace

Vision Vision::load(std::shared_ptr<const mfq::ModelSource> source,const CommonConfig& config) {
    MFQ_RUNTIME_CHECK(source && config.has_vision() && config.image_token_id>=0 &&
        config.image_token_id<config.vocab && config.vision.hidden%config.vision.n_heads==0 &&
        (config.vision.hidden/config.vision.n_heads)%4==0,"invalid DeepSeek-V4.1 CUDA vision config");
    for(const auto* name:{"vision.patch_embedding.weight","vision.output_norm.weight",
            "vision.aligner.input.weight","vision.aligner.output.weight","vision.special_token.start",
            "vision.special_token.newline","vision.special_token.end"})
        MFQ_RUNTIME_CHECK(has_tensor(*source,name),std::string("DeepSeek-V4.1 vision tensor missing: ")+name);
    return Vision(std::move(source),config);
}

mfq::StepSequence<Tensor> Vision::encode(CudaExecutionContext& execution,Tensor patches,int height,int width) const {
    const auto& config=config_.vision;
    const auto hidden=config.hidden,heads=config.n_heads,dim=hidden/heads;
    const int64_t count=int64_t(height)*width;
    const auto activations=std::size_t(count)*std::size_t(std::max(6*hidden,2*config.intermediate))*sizeof(float)+
        (std::size_t(32)<<20);
    Tensor value;
    {
        Affine patch;
        {HostLoading loading(execution);patch=Affine::load(execution,*source_,"vision.patch_embedding",
            3*config.patch_size*config.patch_size,hidden);}
        value=use_weights(execution,std::move(patch),activations,[&](const auto& weight){return weight(execution,patches);});
    }
    co_yield mfq::StepState::advanced;
    std::vector<int32_t> positions;positions.reserve(std::size_t(count)*2);
    for(int row=0;row<height;++row)for(int column=0;column<width;++column){positions.push_back(row);positions.push_back(column);}
    for(int64_t i=0;i<config.n_layers;++i) {
        BlockWeights block;
        {
            HostLoading loading(execution);const auto prefix="vision.block."+std::to_string(i);
            block.norm1=vector_weight(execution,*source_,prefix+".norm1.weight",hidden);
            block.norm2=vector_weight(execution,*source_,prefix+".norm2.weight",hidden);
            block.qkv=Affine::load(execution,*source_,prefix+".attention.qkv",hidden,3*hidden);
            block.output=Affine::load(execution,*source_,prefix+".attention.output",hidden,hidden);
            block.gate_up=Affine::load(execution,*source_,prefix+".mlp.gate_up",hidden,2*config.intermediate);
            block.down=Affine::load(execution,*source_,prefix+".mlp.down",config.intermediate,hidden);
        }
        value=use_weights(execution,std::move(block),activations,[&](const auto& weights) {
            auto qkv=weights.qkv(execution,norm(value,weights.norm1));
            const auto rope=[&](int index){return memory::apply_axial_rope(
                qkv.narrow(-1,index*hidden,hidden).reshape({count,heads,dim}),positions,config.rope_theta);};
            auto query=rope(0),key=rope(1),v=qkv.narrow(-1,2*hidden,hidden).reshape({count,heads,dim});
            auto attended=memory::segmented_attention(query,key,v,{int32_t(count)}).reshape({count,hidden});
            auto result=value+weights.output(execution,std::move(attended));
            auto gate_up=weights.gate_up(execution,norm(result,weights.norm2));
            return result+weights.down(execution,mfq_tensor_backend::silu(gate_up.narrow(-1,0,config.intermediate))*
                                      gate_up.narrow(-1,config.intermediate,config.intermediate));
        });
        co_yield mfq::StepState::advanced;
    }
    {
        const auto ratio=config.downsample_ratio;
        std::vector<int64_t> indices;
        const auto out_h=(height+ratio-1)/ratio,out_w=(width+ratio-1)/ratio;
        for(int64_t row=0;row<out_h;++row)for(int64_t column=0;column<out_w;++column)
            for(int64_t pr=0;pr<ratio;++pr)for(int64_t pc=0;pc<ratio;++pc) {
                const auto r=row*ratio+pr,c=column*ratio+pc;
                indices.push_back(r<height && c<width?r*width+c:count);
            }
        AlignerWeights weights;
        {
            HostLoading loading(execution);
            weights.norm=vector_weight(execution,*source_,"vision.output_norm.weight",hidden);
            weights.input=Affine::load(execution,*source_,"vision.aligner.input",hidden*ratio*ratio,config_.hidden);
            weights.output=Affine::load(execution,*source_,"vision.aligner.output",config_.hidden,config_.hidden);
        }
        value=use_weights(execution,std::move(weights),activations,[&](const auto& device) {
            auto padded=mfq_tensor_backend::cat({norm(value,device.norm),mfq_tensor_backend::zeros({1,hidden},value.options())},0);
            auto index=mfq_tensor_backend::tensor(indices,mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA));
            auto gathered=padded.index_select(0,index).reshape({out_h*out_w,ratio*ratio,hidden})
                .transpose(1,2).contiguous().reshape({out_h*out_w,hidden*ratio*ratio});
            return device.output(execution,mfq_tensor_backend::gelu(device.input(execution,std::move(gathered)),"none"));
        });
    }
    co_yield std::move(value);
}

mfq::StepSequence<CudaPreparedPrompt> Vision::prepare(DeepseekV41CausalLm& language,
        const std::vector<int64_t>& tokens,const MfqMultimodalInput& media) const {
    auto spans=validate_media(config_,tokens,media);
    auto& execution=*language.execution;
    auto options=mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
    auto ids=mfq_tensor_backend::tensor(tokens,options).reshape({1,-1});
    auto embeddings=language.embed_forward(ids).contiguous();
    std::vector<uint8_t> mask(tokens.size(),0);
    std::string request_key="deepseek_v41|";
    for(std::size_t i=0;i<spans.size();++i) {
        const auto& span=spans[i];
        const auto patch_width=media.pixel_shape[2];
        const auto* first=media.pixel_values.data()+int64_t(i)*media.pixel_shape[1]*patch_width;
        auto key=mfq::cache::block_hash_hex(mfq::cache::sha256(first,
            std::size_t(span.height*span.width*patch_width)*sizeof(float)))+"|"+
            std::to_string(span.height)+"|"+std::to_string(span.width);
        Tensor visual;
        if(cached_key_==key && cached_image_.defined())visual=cached_image_;
        else {
            auto patches=mfq_tensor_backend::from_blob(const_cast<float*>(first),{span.height*span.width,patch_width},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU))
                .to(mfq_tensor_backend::kCUDA).to(mfq_tensor_backend::kFloat16);
            auto encoding=encode(execution,std::move(patches),int(span.height),int(span.width));
            while(auto step=encoding.next()){if(step.value)visual=std::move(*step.value);else co_yield step.state;}
            cached_key_=key;cached_image_=visual;
        }
        MFQ_RUNTIME_CHECK(visual.dim()==2 && visual.size(0)==span.rows*span.columns && visual.size(1)==config_.hidden,
                          "DeepSeek-V4.1 aligned image shape mismatch");
        Tensor start,newline,end;
        {HostLoading loading(execution);
            start=vector_weight(execution,*source_,"vision.special_token.start",config_.hidden).to(mfq_tensor_backend::kCUDA);
            newline=vector_weight(execution,*source_,"vision.special_token.newline",config_.hidden).to(mfq_tensor_backend::kCUDA);
            end=vector_weight(execution,*source_,"vision.special_token.end",config_.hidden).to(mfq_tensor_backend::kCUDA);}
        std::vector<Tensor> rows{start.reshape({1,-1}).to(visual.scalar_type())};
        for(int64_t row=0;row<span.rows;++row) {
            rows.push_back(visual.narrow(0,row*span.columns,span.columns));
            rows.push_back(newline.reshape({1,-1}).to(visual.scalar_type()));
        }
        rows.push_back(end.reshape({1,-1}).to(visual.scalar_type()));
        embeddings.narrow(1,span.begin,span.end-span.begin).copy_(
            mfq_tensor_backend::cat(rows,0).unsqueeze(0).to(embeddings.scalar_type()));
        std::fill(mask.begin()+span.begin,mask.begin()+span.end,1);
        request_key+=key+"|"+std::to_string(span.begin)+"|"+std::to_string(span.end)+";";
        co_yield mfq::StepState::advanced;
    }
    CudaPreparedPrompt prepared{tokens,std::move(embeddings),{},0,std::move(request_key)};
    prepared.image_mask=mfq_tensor_backend::tensor(mask,mfq_tensor_backend::TensorOptions()
        .dtype(mfq_tensor_backend::kUInt8).device(mfq_tensor_backend::kCPU)).reshape({1,-1});
    static std::atomic_flag reported=ATOMIC_FLAG_INIT;
    if(!reported.test_and_set())std::fprintf(stderr,"deepseek_v41_vision_dispatch enabled=1 images=%zu weights=streamed engram=text_only router=modality_bias\n",spans.size());
    co_yield std::move(prepared);
}
} // namespace mfq::cuda::deepseek_v41_runtime
