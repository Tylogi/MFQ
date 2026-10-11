#include "weights.h"
#include "e8_dense_group_probe.h"
#include "cpu_projection_rows.h"
#include <chrono>
#include <set>
#include <cstdlib>
#include <climits>

constexpr int routes=10;
#include "e8_dense_group_pack.h"
#include "nvq_dense_group_pack_fast.h"

static std::vector<uint8_t> fast_pack_dense(const NvqWeight& w) {
    const int rows=int(w.out),pairs=int((w.neuron_len+7)/8),groups=int(w.ng);
#define PACK(B,P) nvq_dense_probe::pack<B,P>(w.indices_packed.data_ptr<uint8_t>(),w.indices_packed.numel(), \
        w.aux_packed.data_ptr<uint8_t>(),w.aux_packed.numel(),w.sub_scale_packed.data_ptr<uint8_t>(), \
        w.sub_scale_packed.numel(),rows,pairs,groups)
    switch(w.kernel_format) {
        case 5:return PACK(8,1);case 13:return PACK(10,1);case 14:return PACK(12,1);
        case 10:case 11:return PACK(8,2);case 12:return PACK(9,2);case 15:return PACK(10,2);
        default:throw std::runtime_error("unsupported fast dense pack");
    }
#undef PACK
}

static std::vector<MixedMoePool> collect(const fb::Case& c,const mfq::ModelSource& model) {
    std::vector<MixedMoePool> result;std::set<std::string> visited;
    int format=0,sign_mode=0;
    const auto scan=[&](const std::string& name) {
        if(result.size()==routes || !visited.insert(name).second)return;
        const auto* record=model.find_tensor(name);if(!record)return;
        auto reader=model.tensor_reader(name);
        auto store=std::make_shared<mfq::MfeQuantExpertStore>(size_t(record->nbytes),
            [reader](size_t offset,uint8_t* data,size_t bytes){reader(offset,reinterpret_cast<std::byte*>(data),bytes);});
        MoeQuantRangeSource source(store);
        if(!format){auto w=source.read_expert(c.expert);format=int(w.nvq.kernel_format);sign_mode=int(w.nvq.sign_mode);}
        for(size_t pool=0;pool<store->pool_count() && result.size()<routes;++pool) {
            const auto& w=source.metadata()->pools[pool];
            if(w.family!=MixedMoeFamily::Nvq || w.nvq.kernel_format!=format || w.nvq.sign_mode!=sign_mode)continue;
            for(int expert:store->pool_expert_ids(pool)) {
                auto weight=source.read_expert(expert);
                if(weight.nvq.out!=c.n || weight.nvq.neuron_len!=c.k)throw std::runtime_error("E8 cohort shape");
                result.push_back(std::move(weight));
                std::cout<<"EXPERT "<<c.name()<<'\t'<<name<<'\t'<<expert<<'\n';
                if(result.size()==routes)break;
            }
        }
    };
    scan(c.tensor);
    for(int layer=0;layer<48 && result.size()<routes;++layer)
        for(const auto* role:c.k==640?std::vector<const char*>{"down"}:std::vector<const char*>{"gate","up"})
            scan("model.block."+std::to_string(layer)+".mlp.experts."+role+".weight");
    if(result.size()!=routes)throw std::runtime_error("fewer than ten E8 experts");
    return result;
}
struct Bank {
    std::vector<MixedMoePool> canonical;
    std::vector<tb::Tensor> dense;
    tb::Tensor descriptors;
    Bank(const std::vector<MixedMoePool>& source,const std::vector<std::vector<uint8_t>>& packed,
            const Bank* books=nullptr,int prefix=0) {
        std::vector<E8DenseProbeWeight> weights;
        std::map<const void*,tb::Tensor> shared_books;
        for(size_t i=0;i<source.size();++i) {
            auto& shared=shared_books[source[i].nvq.codebook.data_ptr()];
            if(!books && !shared.defined()) {
                const auto& book=source[i].nvq.codebook;
                const auto* metadata=book.data_ptr<int8_t>();
                // Both controls use the same prefixed allocation and table
                // alignment. Only decoder 4 reads these 16 prepared words.
                std::vector<int8_t> prefixed(64+book.numel());
                std::memcpy(prefixed.data()+64,metadata,book.numel());
                for(int state=0;state<16;++state) {
                    uint16_t scale;std::memcpy(&scale,metadata+4+state*2,2);
                    const uint32_t word=uint32_t(scale)|(uint32_t(uint8_t(metadata[36+state]))<<16);
                    std::memcpy(prefixed.data()+state*4,&word,4);
                }
                shared=tb::tensor(prefixed).to(tb::kCUDA).narrow(0,64,book.numel());
            }
            canonical.push_back(upload(source[i],books?books->canonical[i].nvq.codebook:shared));
            shared=canonical.back().nvq.codebook;
            if(prefix) {
                std::vector<uint8_t> unaligned(prefix,0xa5);unaligned.insert(unaligned.end(),packed[i].begin(),packed[i].end());
                dense.push_back(tb::tensor(unaligned).to(tb::kCUDA).narrow(0,prefix,packed[i].size()));
            }else dense.push_back(tb::tensor(packed[i]).to(tb::kCUDA));
            const auto& w=canonical.back().nvq;E8DenseProbeWeight d;
            d.indices=w.indices_packed.data_ptr<uint8_t>();d.signs=w.aux_packed.data_ptr<uint8_t>();
            d.states=w.sub_scale_packed.data_ptr<uint8_t>();d.dense=dense.back().data_ptr<uint8_t>();
            d.anchors=w.neuron_scale.data_ptr<float>();d.codebook=w.codebook.data_ptr<int8_t>();
            d.index_bytes=w.indices_packed.numel();d.sign_bytes=w.aux_packed.numel();d.dense_bytes=packed[i].size();
            d.rows=int(w.out);d.vectors=int((w.neuron_len+7)/8);d.groups=int(w.ng);d.sign_mode=int(w.sign_mode);
            const int parts=dense_d4_format(int(w.kernel_format))?2:1;
            if(parts==2 && (w.neuron_len+3)/4!=int64_t(d.vectors)*2)
                throw std::runtime_error("D4 dense probe requires paired indices");
            if(int64_t(d.rows)*d.vectors*parts*index_bits(int(w.kernel_format))>INT_MAX-64 ||
                    int64_t(d.rows)*d.vectors*7>INT_MAX-64 || int64_t(d.rows)*d.groups>INT_MAX)
                throw std::runtime_error("E8 probe geometry exceeds narrow decoder contract");
            const int64_t dense_row_bits=int64_t(d.vectors)*(parts*index_bits(int(w.kernel_format))+7)+int64_t(d.groups)*4;
            if(dense_row_bits>INT_MAX || int64_t(d.rows)*dense_row_bits>INT_MAX-64 || d.dense_bytes>INT_MAX)
                throw std::runtime_error("dense probe geometry exceeds local bit-offset contract");
            const auto* meta=source[i].nvq.codebook.data_ptr<int8_t>();
            for(int state=0;state<16;++state) {
                const auto bank=uint8_t(meta[36+state]);
                if(bank>3)throw std::runtime_error("invalid JSC bank map");
                d.bank_map|=uint32_t(bank)<<(2*state);
            }
            weights.push_back(d);
        }
        std::vector<uint8_t> bytes(weights.size()*sizeof(E8DenseProbeWeight));
        std::memcpy(bytes.data(),weights.data(),bytes.size());descriptors=tb::tensor(bytes).to(tb::kCUDA);
    }
    const E8DenseProbeWeight* data()const {return static_cast<const E8DenseProbeWeight*>(descriptors.data_ptr());}
};

static size_t cpu_dense_checks(const NvqWeight& w,const std::vector<uint8_t>& packed,int prefix=0) {
    const auto reference=make_cpu_projection_rows(w);
    const auto dense_kernel=mfq::cpu::nvq_dense_rows_dot_kernel(int(w.kernel_format));
    if(!reference.nvq_rows || !dense_kernel)throw std::runtime_error("dense CPU check requires AVX2 row decoders");
    std::vector<uint8_t> storage(prefix+packed.size());
    std::copy(packed.begin(),packed.end(),storage.begin()+prefix);
    auto dense=reference.nvq;dense.indices=storage.data()+prefix;dense.index_bytes=int64_t(packed.size());
    dense.aux=nullptr;dense.aux_bytes=0;dense.states=nullptr;dense.state_bytes=0;
    auto converted=w;
    const bool converted_ok=prepare_nvq_dense_groups(converted);
    const int bits=index_bits(int(w.kernel_format)),parts=dense_d4_format(int(w.kernel_format))?2:1;
    const bool aligned=(uint64_t(w.out)*(uint64_t((w.neuron_len+7)/8)*(parts*bits+7)+uint64_t(w.ng)*4)&7u)==0;
    if(converted_ok!=aligned)throw std::runtime_error("compact CPU loader expert-boundary selection differs");
    CpuProjectionRows converted_rows;
    if(converted_ok) {
        if(!converted.dense_groups || converted.indices_packed.numel()!=int64_t(packed.size()) ||
           converted.aux_packed.numel()!=0 || converted.sub_scale_packed.numel()!=0 ||
           std::memcmp(converted.indices_packed.data_ptr(),packed.data(),packed.size()))
            throw std::runtime_error("compact CPU loader differs from bit oracle");
        converted_rows=make_cpu_projection_rows(converted);
    }
    size_t checked=0;
    for(int batch:{1,3,8}) {
        const int stride=int(w.neuron_len)+5,out_stride=int(w.out)+3;
        std::vector<float> input(batch*stride);
        for(size_t i=0;i<input.size();++i)input[i]=float(int((i*37+batch*19)%255)-127)*.013f;
        std::vector<float> expected(batch*out_stride,123.0f),actual=expected;
        reference.run(input.data(),batch,stride,expected.data(),out_stride,0,int(w.out));
        dense_kernel(dense,input.data(),batch,stride,actual.data(),out_stride,0,1);
        dense_kernel(dense,input.data(),batch,stride,actual.data(),out_stride,1,w.out-1);
        dense_kernel(dense,input.data(),batch,stride,actual.data(),out_stride,w.out-1,w.out);
        if(std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(float)))
            throw std::runtime_error("compact CPU row bits differ: format="+std::to_string(w.kernel_format)+
                " width="+std::to_string(w.neuron_len)+" batch="+std::to_string(batch));
        if(converted_ok) {
            std::fill(actual.begin(),actual.end(),123.0f);
            converted_rows.run(input.data(),batch,stride,actual.data(),out_stride,0,int(w.out));
            if(std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(float)))
                throw std::runtime_error("compact CPU production rows differ");
        }
        checked+=size_t(batch)*w.out;
    }
    return checked;
}

static size_t routed_dense_checks(const std::vector<MixedMoePool>& sources,
        const std::vector<std::vector<uint8_t>>& packed_streams) {
    const char* option=std::getenv("MFQ_BENCH_DENSE_ROUTED_CHECK");
    if(!option || option[0]=='0')return 0;
    // The canonical 32-lane grouped reference is selected by disabling the
    // optional reuse geometry in this isolated check's environment.
    const char* rows=std::getenv("MFQ_NVQ_ROUTE_ROWS");
    if(!rows || std::string(rows)!="3")throw std::runtime_error("routed dense check requires MFQ_NVQ_ROUTE_ROWS=3");
    const auto& source=sources.at(0);const auto& second=sources.at(1);
    if(source.nvq.codebook.numel()!=second.nvq.codebook.numel() ||
       std::memcmp(source.nvq.codebook.data_ptr(),second.nvq.codebook.data_ptr(),source.nvq.codebook.numel()))
        throw std::runtime_error("multi-expert compact check requires shared cohort codebook");
    auto combined=source;auto other=second;
    auto destination_fields=fields(combined),other_fields=fields(other);
    for(size_t i=0;i<4;++i)*destination_fields[i]=tb::cat({*destination_fields[i],*other_fields[i]},0);
    combined.local_experts=2;combined.nvq.out*=2;combined.nvq.shape[0]=combined.nvq.out;
    const auto original=upload(combined);auto compact=original;
    auto packed=packed_streams.at(0);
    packed.insert(packed.end(),packed_streams.at(1).begin(),packed_streams.at(1).end());
    compact.nvq.indices_packed=tb::tensor(packed).to(tb::kCUDA);
    compact.nvq.aux_packed=tb::empty({0},compact.nvq.indices_packed.options());
    compact.nvq.sub_scale_packed=tb::empty({0},compact.nvq.indices_packed.options());
    const int width=int(source.nvq.neuron_len),outputs=int(source.nvq.out),groups=int(source.nvq.ng);
    auto local=tb::tensor(std::vector<int32_t>{1,0,-1}).to(tb::kCUDA);
    size_t comparisons=0;
    for(bool f16:{false,true})for(bool down:{false,true})for(int batch:(f16?std::vector<int>{9,17,64}:std::vector<int>{1,3,8})) {
        constexpr int routes=5;
        std::vector<int32_t> ids(batch*routes);
        for(size_t i=0;i<ids.size();++i)ids[i]=(i%5==3)?2:(i%3?0:1);
        auto device_ids=tb::tensor(ids).reshape({batch,routes}).to(tb::kCUDA);
        const auto route=build_moe_route_plan(device_ids,3);
        const int input_rows=down?batch*routes:batch;
        auto x=tb::tensor(fb::input(input_rows,width)).reshape({input_rows,width}).to(tb::kCUDA,tb::kFloat16);
        if(down)x=x.reshape({batch,routes,width});
        auto qx=tb::empty({input_rows,groups*24},x.options().dtype(tb::kInt8));
        auto scales=tb::empty({input_rows,groups},x.options().dtype(tb::kFloat32));
        auto expected=tb::full({batch,routes,outputs},123.0f,x.options()),actual=expected.clone();
        const auto call=[&](const NvqWeight& w,tb::Tensor output) {
            if(f16)nvq_moe_grouped_matmul_pool_f16_cuda(w.indices_packed,w.aux_packed,w.sub_scale_packed,
                w.neuron_scale,w.codebook,x,local,3,2,outputs,width,w.gs,w.sub_bits,w.kernel_format,
                w.sign_mode,output,route.ids_dst,route.expert_bounds,route.tile_bounds,route.tile_experts);
            else nvq_moe_grouped_matmul_pool_ws_cuda(w.indices_packed,w.aux_packed,w.sub_scale_packed,
                w.neuron_scale,w.codebook,x,device_ids,local,3,2,outputs,width,w.gs,w.sub_bits,
                w.kernel_format,w.sign_mode,false,output,qx,scales,route.ids_dst,route.expert_bounds,
                route.tile_bounds,route.tile_experts);
        };
        call(original.nvq,expected);call(compact.nvq,actual);
        auto a=expected.to(tb::kCPU),b=actual.to(tb::kCPU);
        if(std::memcmp(a.data_ptr(),b.data_ptr(),a.numel()*a.element_size()))
            throw std::runtime_error("compact routed output bits differ: format="+std::to_string(source.nvq.kernel_format)+
                " batch="+std::to_string(batch)+" f16="+std::to_string(f16)+" down="+std::to_string(down));
        comparisons+=a.numel();
    }
    return comparisons;
}

static void analyze_codebook(const fb::Case& c,const NvqWeight& w) {
    const auto* bytes=w.codebook.data_ptr<int8_t>();
    const int dimensions=dense_d4_format(int(w.kernel_format))?4:8;
    const int entries=1<<index_bits(int(w.kernel_format)),banks=int(uint8_t(bytes[1]));
    if(w.codebook.numel()!=64+int64_t(entries)*dimensions*banks)
        throw std::runtime_error("JSC codebook analysis geometry");
    std::set<int> global;int max_coordinate_unique=0;
    std::cout<<"BOOK_ANALYSIS "<<c.name()<<" banks="<<banks<<" bytes="<<w.codebook.numel()<<" bank_unique=";
    for(int bank=0;bank<banks;++bank) {
        std::set<int> bank_values;
        for(int dimension=0;dimension<dimensions;++dimension) {
            std::set<int> values;
            for(int entry=0;entry<entries;++entry)values.insert(bytes[64+(bank*entries+entry)*dimensions+dimension]);
            max_coordinate_unique=std::max(max_coordinate_unique,int(values.size()));
            bank_values.insert(values.begin(),values.end());
        }
        global.insert(bank_values.begin(),bank_values.end());
        if(bank)std::cout<<',';std::cout<<bank_values.size();
    }
    std::cout<<" global_unique="<<global.size()<<" max_coordinate_unique="<<max_coordinate_unique<<'\n';
}

static void edge_checks(const MixedMoePool& reference,cudaStream_t stream) {
    const int format=int(reference.nvq.kernel_format),bits=index_bits(format);size_t cases=0,float_checks=0;
    const bool d4=dense_d4_format(format);size_t cpu_checks=0;
    if(d4) {
        auto odd=reference.nvq;odd.out=1;odd.neuron_len=4;odd.ng=1;
        bool rejected=false;
        try{pack_dense(odd);}catch(const std::runtime_error&){rejected=true;}
        if(!rejected)throw std::runtime_error("D4 odd-vector contract was not checked");
    }
    auto errors=tb::zeros({1},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
    for(int vectors:{1,2,3,4,5,7,8,9,31,32,33,65})for(int sign_mode:{0,1})for(int prefix:{0,1,2,3}) {
        auto source=reference;auto& w=source.nvq;w.out=17;w.neuron_len=vectors*8-3;
        w.ng=(w.neuron_len+23)/24;w.shape={w.out,w.neuron_len};w.sign_mode=sign_mode;
        const auto payload=[&](size_t bytes,int seed) {
            std::vector<uint8_t> data(bytes);
            for(size_t i=0;i<bytes;++i)data[i]=uint8_t(i*137+(i>>3)*29+seed);
            return tb::tensor(data);
        };
        w.indices_packed=payload((17*vectors*(d4?2:1)*bits+7)/8,17);
        w.aux_packed=payload((17*vectors*7+7)/8,193);
        w.sub_scale_packed=payload((17*w.ng*4+7)/8,83);
        w.neuron_scale=tb::tensor(std::vector<float>(17,.002f));
        const auto packed=pack_dense(w);
        if(fast_pack_dense(w)!=packed)throw std::runtime_error("fast edge pack differs from scalar oracle");
        cpu_checks+=cpu_dense_checks(w,packed,prefix);
        Bank bank({source},{packed},nullptr,prefix);
        e8_dense_probe_validate(format,bank.data(),1,17,static_cast<uint32_t*>(errors.data_ptr()),stream);
        std::vector<int8_t> activation(w.ng*24,0);
        std::vector<float> input_scales(w.ng);
        for(int i=0;i<w.neuron_len;++i)activation[i]=int8_t((i*37+prefix*17+sign_mode*11)%255-127);
        for(int i=0;i<w.ng;++i)input_scales[i]=.003f*float(1+i%9);
        auto x=tb::tensor(activation).to(tb::kCUDA),s=tb::tensor(input_scales).to(tb::kCUDA);
        auto output=tb::zeros({17},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32));
        for(int lanes:{8,16}) {
            e8_dense_probe_launch(format,false,lanes,bank.data(),1,17,x.data_ptr<int8_t>(),s.data_ptr<float>(),output.data_ptr<float>(),stream);
            auto expected=output.to(tb::kCPU);
            for(int dense:{0,1})for(int decoder:{0,1,2,3,4,5,6,7,8,9,10}) {
                if(!dense && !decoder)continue;
                if(d4 && decoder!=0 && decoder!=6)continue;
                e8_dense_probe_launch(format,dense!=0,lanes,bank.data(),1,17,x.data_ptr<int8_t>(),s.data_ptr<float>(),output.data_ptr<float>(),stream,decoder);
                auto actual=output.to(tb::kCPU);
                if(std::memcmp(expected.data_ptr(),actual.data_ptr(),17*sizeof(float)))
                    throw std::runtime_error("E8 edge dot differs: vectors="+std::to_string(vectors)+
                        " prefix="+std::to_string(prefix)+" decoder="+std::to_string(decoder));
                float_checks+=17;
            }
        }
        fb::check(cudaStreamSynchronize(stream));++cases;
    }
    if(errors.to(tb::kCPU).data_ptr<int32_t>()[0])throw std::runtime_error("dense E8 edge decoder differs");
    std::cout<<"EDGE format="<<format<<" cases="<<cases<<" prefixes=0/1/2/3 sign_modes=0/1 float_checks="
        <<float_checks<<" cpu_float_checks="<<cpu_checks<<" exact=1\n"<<std::flush;
}

static void validate(const fb::Case& c,const std::vector<MixedMoePool>& source,
        std::vector<std::unique_ptr<Bank>>& banks,cudaStream_t stream) {
    const int format=int(source[0].nvq.kernel_format),groups=int(source[0].nvq.ng);
    const bool d4=dense_d4_format(format);
    auto errors=tb::zeros({1},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
    e8_dense_probe_validate(format,banks[0]->data(),routes,c.n,static_cast<uint32_t*>(errors.data_ptr()),stream);
    fb::check(cudaStreamSynchronize(stream));
    if(errors.to(tb::kCPU).data_ptr<int32_t>()[0])throw std::runtime_error("dense E8 GPU decode differs");
    std::size_t comparisons=0;
    for(int seed=0;seed<3;++seed) {
        std::vector<int8_t> input(routes*groups*24);
        std::vector<float> scales(routes*groups);
        for(int route=0;route<routes;++route)for(int group=0;group<groups;++group) {
            scales[route*groups+group]=.002f*float(1+(group+seed)%9);
            for(int i=0;i<24;++i)if(group*24+i<c.k)
                input[(route*groups+group)*24+i]=seed?int8_t(((route*137+group*24+i)*37+seed*17)%255-127):0;
        }
        auto x=tb::tensor(input).to(tb::kCUDA),s=tb::tensor(scales).to(tb::kCUDA);
        auto output=tb::zeros({routes*c.n},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32));
        for(int lanes:{8,16}) {
            e8_dense_probe_launch(format,false,lanes,banks[0]->data(),routes,c.n,x.data_ptr<int8_t>(),s.data_ptr<float>(),output.data_ptr<float>(),stream);
            auto expected=output.to(tb::kCPU);
            for(const auto& bank:banks)for(int dense:{0,1})for(int decoder:{0,1,2,3,4,5,6,7,8,9,10}) {
                if(!dense && !decoder)continue;
                if(d4 && decoder!=0 && decoder!=6)continue;
                e8_dense_probe_launch(format,dense!=0,lanes,bank->data(),routes,c.n,x.data_ptr<int8_t>(),s.data_ptr<float>(),output.data_ptr<float>(),stream,decoder);
                auto actual=output.to(tb::kCPU);
                if(std::memcmp(expected.data_ptr(),actual.data_ptr(),actual.numel()*4))
                    throw std::runtime_error("E8 decoder changed MFE pair/quad dot bits");
                comparisons+=actual.numel();
            }
        }
    }
    std::cout<<"VERIFY "<<c.name()<<" decoded_vectors="<<uint64_t(routes)*c.n*((c.k+(d4?3:7))/(d4?4:8))
        <<" float_comparisons="<<comparisons<<" exact=1\n"<<std::flush;
}

int main(int argc,char** argv)try {
    if(argc<2 || argc>3)throw std::runtime_error("usage: mfq-cuda-e8-dense-probe cases.tsv [case-filter]");
    auto context=mfq::cuda::default_context(0);
    auto owned_stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(owned_stream);
    const auto stream=owned_stream.stream();
    cudaDeviceProp properties{};fb::check(cudaGetDeviceProperties(&properties,0));
    std::map<std::string,std::shared_ptr<const mfq::ModelSource>> models;
    std::set<int> edge_formats;
    for(const auto& c:fb::read_cases(argv[1])) {
        if((c.label.find("NVQ-E8-")!=0 && c.label.find("NVQ-D4-")!=0) ||
                (argc==3 && c.name().find(argv[2])==std::string::npos))continue;
        auto& model=models[c.model];if(!model)model=mfq::open_model_source(c.model);
        auto source=collect(c,*model);std::vector<std::vector<uint8_t>> packed;
        analyze_codebook(c,source[0].nvq);
        if(edge_formats.insert(int(source[0].nvq.kernel_format)).second)edge_checks(source[0],stream);
        if(source[0].nvq.kernel_format==11 && edge_formats.insert(10).second) {
            // The real 256-entry cohorts use the two-bank analytic kernel.
            // Also exercise the generic four-bank LUT decoder independently.
            auto generic=source[0];generic.nvq.kernel_format=10;
            std::vector<int8_t> metadata(64+4*256*4);
            metadata[0]=1;metadata[1]=4;metadata[2]=16;
            for(int state=0;state<16;++state) {
                const uint16_t half=uint16_t(0x3000+state*0x40);
                metadata[4+2*state]=int8_t(half&255);metadata[5+2*state]=int8_t(half>>8);
                metadata[36+state]=int8_t((state*3+1)&3);
            }
            for(size_t i=64;i<metadata.size();++i)metadata[i]=int8_t((i*7+(i>>3)*11)&127);
            generic.nvq.codebook=tb::tensor(metadata);
            edge_checks(generic,stream);
        }
        size_t source_bytes=0,dense_bytes=0,cpu_checks=0,routed_checks=0;double fast_pack_ms=0;
        const auto begin=std::chrono::steady_clock::now();
        for(auto& expert:source) {
            packed.push_back(pack_dense(expert.nvq));source_bytes+=storage(expert);
            dense_bytes+=packed.back().size()+expert.nvq.neuron_scale.numel()*sizeof(float);
            const auto fast_begin=std::chrono::steady_clock::now();
            const auto fast=fast_pack_dense(expert.nvq);
            fast_pack_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fast_begin).count();
            if(fast!=packed.back())throw std::runtime_error("fast real pack differs from scalar oracle");
            cpu_checks+=cpu_dense_checks(expert.nvq,packed.back());
        }
        routed_checks=routed_dense_checks(source,packed);
        const double pack_check_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        const size_t count=std::max<size_t>(2,(4*size_t(properties.l2CacheSize)+source_bytes-1)/source_bytes);
        std::vector<std::unique_ptr<Bank>> banks;
        for(size_t i=0;i<count;++i)banks.push_back(std::make_unique<Bank>(source,packed,i?banks.front().get():nullptr));
        validate(c,source,banks,stream);
        std::cout<<"FAST_PACK "<<c.name()<<" bytes="<<dense_bytes<<" ms="<<fast_pack_ms<<" scalar_exact=1\n"<<std::flush;
        std::cout<<"CPU_VERIFY "<<c.name()<<" float_comparisons="<<cpu_checks<<" exact=1\n"<<std::flush;
        if(routed_checks)std::cout<<"ROUTED_VERIFY "<<c.name()<<" half_comparisons="<<routed_checks<<" exact=1\n"<<std::flush;
        if(const char* only=std::getenv("MFQ_BENCH_DENSE_CHECK_ONLY");only && only[0]!='0')continue;
        const int groups=int(source[0].nvq.ng),format=int(source[0].nvq.kernel_format);
        std::vector<int8_t> input(routes*groups*24);std::vector<float> scales(routes*groups,.003f);
        for(int route=0;route<routes;++route)for(int i=0;i<c.k;++i)input[route*groups*24+i]=int8_t((i*37+route*17)%255-127);
        auto x=tb::tensor(input).to(tb::kCUDA),s=tb::tensor(scales).to(tb::kCUDA);
        auto output=tb::zeros({routes*c.n},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32));
        const char* bank_option=std::getenv("MFQ_BENCH_E8_BANK_MAP");
        const bool bank_experiment=!dense_d4_format(format) && bank_option && bank_option[0]!='0';
        const char* state_option=std::getenv("MFQ_BENCH_E8_STATE_WORD");
        const bool state_experiment=!dense_d4_format(format) && state_option && state_option[0]!='0';
        const char* pipeline_option=std::getenv("MFQ_BENCH_E8_PIPELINE");
        const bool pipeline_experiment=!dense_d4_format(format) && pipeline_option && pipeline_option[0]!='0';
        const char* narrow_option=std::getenv("MFQ_BENCH_E8_DENSE_NARROW");
        const bool narrow_experiment=narrow_option && narrow_option[0]!='0';
        const char* cache_option=std::getenv("MFQ_BENCH_E8_CACHE_POLICY");
        const bool cache_experiment=!dense_d4_format(format) && cache_option && cache_option[0]!='0';
        if(int(state_experiment)+int(bank_experiment)+int(pipeline_experiment)+int(narrow_experiment)+int(cache_experiment)>1)
            throw std::runtime_error("choose one decoder experiment");
        const char* decoder_option=std::getenv("MFQ_BENCH_E8_DECODER");
        const int decoder_variant=decoder_option?std::atoi(decoder_option):0;
        if(decoder_variant!=0 && decoder_variant!=2 && decoder_variant!=3)
            throw std::runtime_error("E8 decoder benchmark expects 0, 2 or 3");
        if(dense_d4_format(format) && decoder_variant)
            throw std::runtime_error("D4 decoder benchmark expects 0");
        const std::vector<std::pair<int,int>> variants=cache_experiment && cache_option[0]=='2'?
            std::vector<std::pair<int,int>>{{1,10},{1,7},{1,10}}:cache_experiment?
            std::vector<std::pair<int,int>>{{1,0},{1,7},{1,0},{1,8},{1,0},{1,9},{1,0}}:narrow_experiment?
            std::vector<std::pair<int,int>>{{1,dense_d4_format(format)?0:2},{1,6},{1,dense_d4_format(format)?0:2}}:pipeline_experiment?
            std::vector<std::pair<int,int>>{{1,2},{1,5},{1,2}}:state_experiment?
            std::vector<std::pair<int,int>>{{0,2},{0,4},{0,2},{1,2},{1,4},{1,2}}:decoder_variant?
            std::vector<std::pair<int,int>>{{0,decoder_variant},{1,decoder_variant},{0,decoder_variant}}:bank_experiment?
            std::vector<std::pair<int,int>>{{0,0},{0,1},{0,0},{1,0},{1,1},{1,0}}:
            std::vector<std::pair<int,int>>{{0,0},{1,0},{0,0}};
        for(int lanes:{16,8})for(const auto variant:variants) {
            const int dense=variant.first,decoder=variant.second;
            cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
            fb::check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            for(int repeat=0;repeat<8;++repeat)for(const auto& bank:banks)
                e8_dense_probe_launch(format,dense!=0,lanes,bank->data(),routes,c.n,x.data_ptr<int8_t>(),s.data_ptr<float>(),output.data_ptr<float>(),stream,decoder);
            fb::check(cudaStreamEndCapture(stream,&graph));fb::check(cudaGraphInstantiate(&exec,graph,0));
            const auto samples=fb::measure([&]{fb::check(cudaGraphLaunch(exec,stream));},stream,8*banks.size());
            const auto attributes=e8_dense_probe_attributes(format,dense!=0,lanes,decoder);
            const auto us=fb::median(samples);const auto bytes=dense?dense_bytes:source_bytes;
            std::cout<<std::setprecision(10)<<"{\"case\":\""<<c.name()<<"\",\"dense\":"<<dense
                <<",\"lanes\":"<<lanes<<",\"decoder\":"<<decoder<<",\"bank_map\":"<<(decoder==1)
                <<",\"experts\":10,\"banks\":"<<count<<",\"weight_bytes\":"<<bytes
                <<",\"source_bytes\":"<<source_bytes<<",\"dense_bytes\":"<<dense_bytes<<",\"us\":"<<us
                <<",\"effective_GBs\":"<<bytes/(us*1000.)<<",\"regs\":"<<attributes.numRegs
                <<",\"local_bytes\":"<<attributes.localSizeBytes
                <<",\"pack_and_cpu_check_ms\":"<<pack_check_ms<<",\"scope\":\"projection_only_prequantized_input\",\"samples_us\":[";
            for(size_t i=0;i<samples.size();++i)std::cout<<(i?",":"")<<samples[i];
            std::cout<<"]}\n"<<std::flush;
            fb::check(cudaGraphExecDestroy(exec));fb::check(cudaGraphDestroy(graph));
        }
    }
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
