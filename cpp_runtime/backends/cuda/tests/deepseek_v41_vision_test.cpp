#include "models/deepseek_v41/vision.h"
#include <iostream>

namespace {
namespace tb=mfq_tensor_backend;
using namespace mfq::cuda;
using namespace mfq::cuda::deepseek_v41_runtime;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}

struct Source final:mfq::ModelSource {
    std::vector<mfq::TensorMetadata> records;
    std::unordered_map<std::string,std::vector<std::byte>> blobs;
    mutable int reads=0;
    void add(std::string name,std::vector<int64_t> shape,std::vector<float> values) {
        uint32_t rank=uint32_t(shape.size());std::vector<std::byte> blob(4+8*shape.size()+4*values.size());
        std::memcpy(blob.data(),&rank,4);std::memcpy(blob.data()+4,shape.data(),8*shape.size());
        std::memcpy(blob.data()+4+8*shape.size(),values.data(),4*values.size());
        records.push_back({name,"F32","F32",shape,blob.size()});blobs.emplace(std::move(name),std::move(blob));
    }
    void matrix(const std::string& prefix,int out,int in,bool bias=true) {
        add(prefix+".weight",{out,in},std::vector<float>(out*in));
        if(bias)add(prefix+".bias",{out},std::vector<float>(out));
    }
    const std::vector<std::filesystem::path>& source_paths() const noexcept override {
        static const std::vector<std::filesystem::path> paths;return paths;
    }
    std::string_view architecture() const noexcept override{return "deepseek_v41";}
    const std::unordered_map<std::string,std::string>& metadata() const noexcept override {
        static const std::unordered_map<std::string,std::string> values;return values;
    }
    const std::vector<mfq::TensorMetadata>& tensors() const noexcept override{return records;}
    const mfq::TensorMetadata* find_tensor(std::string_view name) const noexcept override {
        for(const auto& record:records)if(record.name==name)return &record;return nullptr;
    }
    void read_range_into(std::string_view name,uint64_t offset,std::byte* target,std::size_t bytes) const override {
        const auto& data=blobs.at(std::string(name));
        check(offset<=data.size() && bytes<=data.size()-offset,"vision fixture read is out of bounds");
        ++reads;std::memcpy(target,data.data()+offset,bytes);
    }
    const std::vector<std::string>& assets() const noexcept override{static const std::vector<std::string> values;return values;}
    bool has_asset(std::string_view) const noexcept override{return false;}
    std::vector<std::byte> read_asset(std::string_view) const override{return {};}
    std::optional<mfq::ModelGraph> model_graph() const override{return {};}
};

std::shared_ptr<Source> fixture() {
    auto s=std::make_shared<Source>();
    s->add("vision.patch_embedding.weight",{8,12},std::vector<float>(8*12));
    std::vector<float> patch(8);for(int i=0;i<8;++i)patch[i]=.1f*(i+1);
    s->add("vision.patch_embedding.bias",{8},patch);
    s->add("vision.output_norm.weight",{8},std::vector<float>(8,1.f));
    for(auto name:{"norm1","norm2"})s->add(std::string("vision.block.0.")+name+".weight",{8},std::vector<float>(8,1.f));
    s->matrix("vision.block.0.attention.qkv",24,8);s->matrix("vision.block.0.attention.output",8,8);
    s->matrix("vision.block.0.mlp.gate_up",24,8,false);s->matrix("vision.block.0.mlp.down",8,12,false);
    std::vector<float> up(16*72),down(16*16),bias(16);
    for(int i=0;i<16;++i)for(int j=0;j<72;++j)up[i*72+j]=float((i*7+j*11)%17)*.001f;
    for(int i=0;i<16;++i)for(int j=0;j<16;++j)down[i*16+j]=float((i+j)%5)*.003f;
    for(int i=0;i<16;++i)bias[i]=float(i)*.003f;
    s->add("vision.aligner.input.weight",{16,72},up);s->add("vision.aligner.input.bias",{16},bias);
    s->add("vision.aligner.output.weight",{16,16},down);s->add("vision.aligner.output.bias",{16},std::vector<float>(16,.2f));
    s->add("vision.special_token.start",{16},std::vector<float>(16,1.f));
    s->add("vision.special_token.newline",{16},std::vector<float>(16,-1.f));
    s->add("vision.special_token.end",{16},std::vector<float>(16,2.f));
    return s;
}

CudaPreparedPrompt run(Vision& vision,DeepseekV41CausalLm& language,
                       const std::vector<int64_t>& ids,const MfqMultimodalInput& media) {
    CudaPreparedPrompt result;auto preparation=vision.prepare(language,ids,media);
    int steps=0;while(auto step=preparation.next()){if(step.value)result=std::move(*step.value);else ++steps;}
    check(result.embeddings.defined() && steps>0,"vision preparation did not yield a prepared prompt");return result;
}

void test() {
    CudaExecutionContext execution;DeepseekV41CausalLm language;language.execution=&execution;
    auto& config=language.config;config.hidden=16;config.vocab=8;config.image_token_id=7;
    config.vision={1,8,1,12,2,3,64,12,10000.};
    language.embed.kind=QuantLinearKind::Dense;
    language.embed.dense=tb::ones({8,16},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16))*.5;
    language.shared=std::make_shared<SharedState>();
    auto source=fixture();auto vision=Vision::load(source,config);
    check(source->reads==0,"text-only loading materialized vision weights");
    MfqMultimodalInput media;media.processor=MfqMultimodalProcessor::deepseek_v41;media.processor_name="deepseek_v41";
    media.pixel_shape={1,20,12};media.pixel_values.assign(240,.3f);media.vision_grid={4,5,2,2};
    media.vision_grid_shape={1,4};media.image_bounds={0,0,1,9};
    std::vector<int64_t> ids{1,7,7,7,7,7,7,7,7,2};
    auto prepared=run(vision,language,ids,media);
    check(!prepared.positions.defined(),"DeepSeek-V4.1 Vision must retain implicit causal cache positions");
    auto actual=prepared.embeddings.to(tb::kCPU,tb::kFloat32).contiguous();
    const auto* data=actual.data_ptr<float>();
    for(int col=0;col<16;++col) {
        check(data[col]==.5f && data[9*16+col]==.5f,"vision replacement modified text embeddings");
        check(data[16+col]==1.f && data[4*16+col]==-1.f && data[7*16+col]==-1.f && data[8*16+col]==2.f,
              "image start/newline/end embedding order changed");
    }
    // Independent CPU reference for RMSNorm, zero padded 3x3 pixel-unshuffle,
    // and both aligner matrices. Zero attention/MLP weights preserve patch bias.
    float patch[8],sum=0;
    for(int channel=0;channel<8;++channel){patch[channel]=float(mfq_half(.1f*(channel+1)));sum+=patch[channel]*patch[channel];}
    for(auto& channel:patch)channel=float(mfq_half(channel/std::sqrt(sum/8+1e-6f)));
    for(int row=0;row<2;++row)for(int column=0;column<2;++column) {
        float hidden[16];
        for(int out=0;out<16;++out) {
            float value=float(mfq_half(out*.003f));
            for(int channel=0;channel<8;++channel)for(int pr=0;pr<3;++pr)for(int pc=0;pc<3;++pc) {
                const int index=channel*9+pr*3+pc;
                if(row*3+pr<4 && column*3+pc<5)value+=patch[channel]*float(mfq_half(float((out*7+index*11)%17)*.001f));
            }
            hidden[out]=.5f*value*(1.f+std::erf(value/std::sqrt(2.f)));
        }
        const int token=2+row*3+column;
        for(int out=0;out<16;++out) {
            float expected=.2f;
            for(int j=0;j<16;++j)expected+=hidden[j]*float(mfq_half(float((out+j)%5)*.003f));
            check(std::abs(data[token*16+out]-expected)<.003f,"vision unshuffle/aligner differs from CPU reference");
        }
    }
    check(prepared.image_mask.is_cpu() && prepared.image_mask.numel()==10,"vision prompt mask storage differs");
    {
        CudaImageMaskScope scope(language,&prepared,0,5);
        check(language.shared->text_row_ids==std::vector<int64_t>{0},"prepared chunk image mask selected image rows for Engram");
    }
    check(!language.shared->image_mask.defined(),"image participation survived the prepared prefill scope");
    const auto reads=source->reads;auto cached=run(vision,language,ids,media);
    check(source->reads-reads==3 && cached.cache_key==prepared.cache_key,"image cache reloaded the vision tower");
    media.image_bounds[3]=8;
    try{run(vision,language,ids,media);throw std::logic_error("malformed image span was accepted");}
    catch(const std::invalid_argument&){}
    media.image_bounds[3]=9;
    media.pixel_values[0]+=.1f;
    {auto cancelled=vision.prepare(language,ids,media);check(bool(cancelled.next()),"vision preparation made no cancellable progress");}
    run(vision,language,ids,media);
    std::cout<<"DeepSeek-V4.1 streamed vision, padding, sentinels, mask, cache and cancellation PASS\n";
}
}
int main() try {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || !count)return 77;
    test();return 0;
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
