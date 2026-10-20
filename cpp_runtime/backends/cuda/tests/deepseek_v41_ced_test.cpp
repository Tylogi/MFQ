#include "models/deepseek_v41/ops.h"
#include "storage/session_state.h"
#include "storage/text_session_cache.h"
#include "runtime_config.h"
#include "mfq_paged_prefix_cache.h"
#include "mfq_cuda_moe_ops.h"
#include <chrono>
#include <filesystem>
#include <iostream>

namespace {
namespace tb=mfq_tensor_backend;
using namespace mfq::cuda;
using namespace mfq::cuda::deepseek_v41_runtime;

void require(bool value,const char* message) {
    if(!value)throw std::runtime_error(message);
}
tb::Tensor fp32(std::vector<int64_t> shape) {
    return tb::ones(shape,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32));
}
QuantLinear dense(int out,int in) {
    QuantLinear linear;
    linear.kind=QuantLinearKind::Dense;
    linear.dense=(tb::randn({out,in},fp32({1}).options())/std::sqrt(double(in)))
        .to(tb::kFloat16).contiguous();
    return linear;
}

void check_tail_rope() {
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    int cases=0;
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int tokens:{1,6,129,4080}) {
        const int heads=tokens==4080?64:3,width=tokens==4080?512:96;
        auto x=tb::full({1,tokens,heads,width},.25,options.dtype(dtype)).transpose(1,2);
        auto positions=tb::arange(tokens,options.dtype(tb::kInt64));
        auto angles=tb::arange(tokens*32,options).reshape({tokens,32})*.013;
        auto cosine=tb::cos(angles),sine=tb::sin(angles);
        auto head=x.slice(-1,0,width-64).contiguous(),tail=x.slice(-1,width-64,width).contiguous();
        auto rotated=glm_interleaved_rope_cuda(tail,positions,cosine,sine,64);
        auto expected=tb::cat({head,rotated},-1).contiguous();
        auto actual=dsv4_tail_rope_cuda(x,positions,cosine,sine,64);
        require((actual.to(tb::kFloat32)-expected.to(tb::kFloat32)).abs().max().item<float>()==0,
            "fused strided tail RoPE differs from materialized rotation");
        ++cases;
    }
    std::cout<<"tail_rope_cases="<<cases<<" FP16/FP32 strided/large-grid exact PASS\n";
}

void check_modality_routing() {
    int cases=0;
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int experts:{384,641})for(int rows:{2,5}) {
        auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
        auto logits=(tb::arange(int64_t(rows)*experts,options)*.001).reshape({rows,experts}).to(dtype).contiguous();
        std::vector<float> biases(rows*experts,0.f);
        for(int row=0;row<rows;++row)biases[row*experts+(row*53+7)%experts]=1000.f;
        auto bias=tb::tensor(biases,options).reshape({rows,experts});
        auto batch=moe_topk_cuda(logits,6,false,true,true,false,bias,1e-20,1.0);
        auto shared_bias=bias.narrow(0,0,1).contiguous();
        auto shared=moe_topk_cuda(logits,6,false,true,true,false,shared_bias,1e-20,1.0);
        auto flat_bias=moe_topk_cuda(logits,6,false,true,true,false,shared_bias.reshape({experts}),1e-20,1.0);
        require((shared[0].to(tb::kFloat32)-flat_bias[0].to(tb::kFloat32)).abs().max().item<float>()==0 &&
            (shared[1]-flat_bias[1]).abs().max().item<float>()<1e-6,
            "per-token routing removed the existing shared correction bias broadcast");
        for(int row=0;row<rows;++row) {
            auto expected=moe_topk_cuda(logits.narrow(0,row,1).contiguous(),6,false,true,true,false,
                bias.narrow(0,row,1).reshape({experts}),1e-20,1.0);
            require((batch[0].narrow(0,row,1).to(tb::kFloat32)-expected[0].to(tb::kFloat32)).abs().max().item<float>()==0,
                    "per-token vision bias selected another token's experts");
            require((batch[1].narrow(0,row,1)-expected[1]).abs().max().item<float>()<1e-6,
                    "vision correction bias changed unbiased routing weights");
        }
        ++cases;
    }
    std::cout<<"modality_router_cases="<<cases<<" cached/generic FP16/FP32 PASS\n";
}

DeepseekV41CausalLm make_model(CudaExecutionContext& execution,int ratio=2) {
    tb::manual_seed(91);
    DeepseekV41CausalLm model;
    model.execution=&execution;
    auto& c=model.config;
    c.hidden=64;c.vocab=8;c.n_layers=4;c.causal_encoder_layers=2;
    c.hc_mult=4;c.hc_sinkhorn_iters=20;c.hc_eps=1.e-6;
    c.n_heads=64;c.n_kv_heads=1;c.head_dim=512;c.rope_head_dim=64;
    c.q_lora_rank=32;c.o_groups=8;c.o_lora_rank=8;
    c.index_n_heads=32;c.index_head_dim=128;c.index_topk=16;
    c.candidate_source_layer=2;c.candidate_block_size=8;c.candidate_topk_blocks=8;
    c.compress_ratios={0,ratio,1,1};c.kv_source_layers={1,2};c.index_source_layers={1,2,3};
    c.sliding_window=128;c.max_position_embeddings=1024;c.dspark_target_layer_ids={2,3};
    model.metadata.hidden_size=c.hidden;model.metadata.vocab_size=c.vocab;
    model.metadata.num_hidden_layers=c.n_layers;model.metadata.hc_mult=4;
    model.metadata.max_position_embeddings=c.max_position_embeddings;
    model.metadata.rms_norm_eps=c.rms_eps;
    model.embed=dense(c.vocab,c.hidden);model.output_norm=fp32({c.hidden});
    model.shared=std::make_shared<SharedState>();model.shared->config=c;
    model.shared->engram_hash=std::make_unique<EngramHashState>(8,3,2,0,
        std::vector<int64_t>{1},std::vector<int64_t>{36},std::vector<int64_t>{5,7,11,13},
        std::vector<int64_t>{0,5,12,23},std::vector<int64_t>{3,5,7},
        std::vector<int>{0,1,2,3,4,5,6,7});
    for(int layer=0;layer<c.n_layers;++layer) {
        auto block=std::make_unique<deepseek_v41_runtime::Block>();
        auto& b=*block;
        b.config=c;b.shared=model.shared;b.layer=layer;b.ratio=c.compress_ratios[layer];
        b.max_context=c.max_position_embeddings;b.attention_norm=b.mlp_norm=fp32({c.hidden});
        b.query_a_norm=fp32({c.q_lora_rank});b.key_value_norm=b.compressor_norm=fp32({512});
        b.index_key_norm=fp32({128});b.sinks=tb::zeros({64},fp32({1}).options());
        b.attention_mhc_function=tb::randn({24,4*c.hidden},fp32({1}).options())*.03;
        b.mlp_mhc_function=tb::randn({24,4*c.hidden},fp32({1}).options())*.03;
        b.attention_mhc_scale=b.mlp_mhc_scale=fp32({3});
        b.attention_mhc_base=b.mlp_mhc_base=tb::zeros({24},fp32({1}).options());
        b.query_a=dense(c.q_lora_rank,c.hidden);b.query_b=dense(64*512,c.q_lora_rank);
        b.key_value=dense(512,c.hidden);b.output_a=dense(8*8,64*512/8);b.output_b=dense(c.hidden,8*8);
        b.compressor_key_value=dense(512,c.hidden);b.compressor_gate=dense(512,c.hidden);
        b.index_key=dense(128,512);b.index_query=dense(32*128,c.q_lora_rank);b.index_score=dense(32,c.hidden);
        b.mlp.gate_up.layers.push_back(dense(128,c.hidden));
        b.mlp.gate_up.layers.push_back(dense(128,c.hidden));b.mlp.down=dense(c.hidden,128);
        b.rope=Dsv4RopeTable(c.max_position_embeddings,c.rope_theta,b.ratio,c.compress_rope_theta);
        model.blocks.push_back(std::move(block));
    }
    model.reset(1);
    return model;
}
tb::Tensor ids(int tokens) {
    return tb::arange(tokens,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64))
        .remainder(8.).to(tb::kInt64).reshape({1,tokens}).contiguous();
}
void check_positions(const DeepseekV41CausalLm& model,int64_t expected) {
    require(model.cache_pos==expected,"CED logical cache position changed");
    for(const auto& block:model.blocks) {
        const auto& b=dynamic_cast<const deepseek_v41_runtime::Block&>(*block);
        require(b.state.position==expected,"CED physical cache position changed");
        if(b.kv_source())require(b.state.compressed_length==expected/b.ratio,
                                "CED duplicated or lost global KV rows");
    }
}

void check_session_snapshots() {
    CudaExecutionContext execution;
    int cases=0;
    for(int ratio:{1,2,16})for(int length:{3,257}) {
        auto model=make_model(execution,ratio);
        deepseek_v41_prefill_hidden(model,ids(length),64);
        std::vector<int64_t> tokens(length);
        for(int i=0;i<length;++i)tokens[i]=i%8;
        auto saved=model.capture_text_session_state(tokens);
        require(saved.kind()==TextSessionStateKind::DeepseekV41 && saved.bytes>0,
                "DeepSeek-V4.1 session support or byte accounting is missing");
        auto expected=model.hidden_forward(ids(3));
        const auto expected_hashes=model.shared->engram_hashes.values;
        model.reset(1);model.hidden_forward(ids(11));
        model.restore_text_session_state(saved);check_positions(model,length);
        auto actual=model.hidden_forward(ids(3));
        require(actual.equal(expected) && model.shared->engram_hashes.values==expected_hashes,
                "DeepSeek-V4.1 restored attention or Engram changed continuation");
        model.restore_text_session_state(saved);
        require(model.hidden_forward(ids(3)).equal(expected),"session restore modified its immutable snapshot");
        auto broken=saved;
        auto& layer=std::get<std::vector<DeepseekV41BlockSessionState>>(broken.payload).front();
        layer.local_kv=layer.local_kv.narrow(1,0,127);
        bool rejected=false;
        try{model.restore_text_session_state(broken);}catch(const CudaSessionStateError&){rejected=true;}
        require(rejected && model.cache_pos==length+3,"invalid session geometry changed the active model");
        auto hash=*saved.deepseek_v41_engram;
        model.shared->engram_hash->restore(hash);
        hash.context.push_back(0);
        rejected=false;
        try{model.shared->engram_hash->restore(hash);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected && model.shared->engram_hash->snapshot().position==length,
                "invalid Engram snapshot changed rolling history");
        ++cases;
    }
    mfq::models::deepseek_v41::EngramHashState hash(8,3,2,0,{1},{36},{5,7,11,13},{0,5,12,23},{3,5,7},{0,1,2,3,4,5,6,7});
    const std::array<int64_t,3> prefix{1,2,3};const std::array<uint8_t,3> image{0,1,0};
    hash.forward(prefix.data(),1,3,0,image.data());
    auto saved=hash.snapshot();const std::array<int64_t,2> suffix{4,5};
    auto expected=hash.forward(suffix.data(),1,2,3);
    hash.restore(saved);
    auto actual=hash.forward(suffix.data(),1,2,3);
    require(expected.values==actual.values && saved.context.front()==mfq::models::deepseek_v41::kDeadToken,
            "Engram session lost an image-span ngram boundary");
    std::cout<<"session_snapshot_cases="<<cases<<" compressor/ring/Engram/immutable/rejection PASS\n";
}
void check_paged_sessions() {
    using mfq::cuda::internal::TextSessionCache;
    CudaExecutionContext execution;
    const auto directory=std::filesystem::temp_directory_path()/
        ("mfq-v41-prefix-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    int cases=0;
    for(int ratio:{1,2,16})for(int length:{3,128,257}) {
        auto model=make_model(execution,ratio);
        deepseek_v41_prefill_hidden(model,ids(length),64);
        std::vector<int64_t> tokens(length);for(int i=0;i<length;++i)tokens[i]=i%8;
        auto saved=model.capture_text_session_state(tokens);
        auto expected=model.hidden_forward(ids(3));
        const auto hashes=model.shared->engram_hashes.values;
        for(size_t block:{64,128}) {
            std::vector<CudaPagedPayload> payloads;
            for(size_t i=0;i<(tokens.size()-1)/block+1;++i)
                payloads.push_back(encode_cuda_deepseek_v41_paged_block(saved,block,i));
            auto decoded=decode_cuda_deepseek_v41_paged_session(payloads,tokens,block);
            require(decoded.bytes==saved.bytes,"paged DeepSeek-V4.1 byte accounting changed");
            model.restore_text_session_state(decoded);
            require(model.hidden_forward(ids(3)).equal(expected) && model.shared->engram_hashes.values==hashes,
                "paged DeepSeek-V4.1 changed continuation or Engram history");
            auto truncated=std::make_shared<std::vector<uint8_t>>(*payloads.back());truncated->pop_back();
            auto broken=payloads;broken.back()=truncated;bool rejected=false;
            try{decode_cuda_deepseek_v41_paged_session(broken,tokens,block);}catch(const CudaSessionStateError&){rejected=true;}
            require(rejected,"truncated DeepSeek-V4.1 prefix was accepted");
            payloads.pop_back();
            require(cuda_paged_decodable_blocks(payloads)==0,"compressed KV without a local/Engram checkpoint was restored");
        }
        // Real tier transitions, including complete RAM eviction with SSD disabled.
        for(int storage:{0,1,2,3}) {
            mfq::cache::PagedPrefixCacheConfig config;
            config.cache_dir=directory/(std::to_string(ratio)+"-"+std::to_string(length)+"-"+std::to_string(storage));
            config.compatibility_key="cuda-v41-prefix-test";config.block_size_tokens=64;
            config.spill_hot_on_eviction=true;config.max_hot_bytes=32ULL<<20;
            config.max_disk_bytes=(storage&1)?64ULL<<20:0;
            auto paged=std::make_shared<mfq::cache::PagedPrefixCache>(config);
            mfq::engine::PrefixCacheConfig prefix;prefix.hot_bytes=0;prefix.ram_bytes=config.max_hot_bytes;
            TextSessionCache cache({},prefix,paged,true,0,true);
            auto state=saved;
            if(storage>=2)state.input_key="image-A";
            cache.store("",state);
            auto prompt=tokens;prompt.push_back(7);
            require(cache.restore_best(model,nullptr,"",prompt,tokens.size(),state.input_key).tokens==tokens.size(),
                "DeepSeek-V4.1 RAM prefix restore failed");
            require(model.hidden_forward(ids(3)).equal(expected),"RAM prefix changed continuation");
            TextSessionCache unified({},prefix,paged,true,0,false);
            require(unified.restore_best(model,nullptr,"",prompt,tokens.size(),state.input_key).tokens==tokens.size(),
                "unified CUDA prefix lookup missed a paged checkpoint");
            require(cache.restore_best(model,nullptr,"",prompt,tokens.size(),"different-image").tokens==0,
                "DeepSeek-V4.1 prefix crossed image identities");
            auto& mtp=state.mtp.emplace();mtp.cache_pos=length;
            mtp.last_target_hidden=tb::ones({1,1,128},fp32({1}).options().dtype(tb::kFloat16));
            mtp.deepseek_v41_rings={tb::ones({1,128,512},mtp.last_target_hidden.options())};
            mtp.bytes=session_tensor_bytes(mtp.last_target_hidden)+session_tensor_bytes(mtp.deepseek_v41_rings.front());
            state.bytes+=mtp.bytes;
            cache.store("",state);
            const auto match=paged->match(tokens,state.input_key,false);
            auto payloads=paged->load_prefix(match.blocks);
            require(!payloads.empty() && !cuda_paged_checkpoint_needs_update(payloads.back(),state),
                "existing prefix was not upgraded with DSpark state");
            auto decoded=decode_cuda_deepseek_v41_paged_session(payloads,tokens,64);
            require(decoded.mtp && decoded.mtp->cache_pos==length &&
                decoded.mtp->last_target_hidden.equal(mtp.last_target_hidden) &&
                decoded.mtp->deepseek_v41_rings.front().equal(mtp.deepseek_v41_rings.front()),
                "paged DSpark state changed");
            cache.set_ram_limit(0);paged->flush();
            require(cache.restore_best(model,nullptr,"",prompt,tokens.size(),state.input_key).tokens==
                ((storage&1)?tokens.size():0),"RAM eviction did not spill or discard DeepSeek-V4.1 prefix");
            if(storage&1) {
                require(model.hidden_forward(ids(3)).equal(expected),"SSD prefix changed continuation");
                config.max_hot_bytes=0;
                auto reopened=std::make_shared<mfq::cache::PagedPrefixCache>(config);
                TextSessionCache restored({},prefix,reopened,true,0,true);
                require(restored.restore_best(model,nullptr,"",prompt,tokens.size(),state.input_key).tokens==tokens.size(),
                    "persisted DeepSeek-V4.1 prefix was not reusable after reopening");
            }
        }
        ++cases;
    }
    std::cout<<"paged_session_cases="<<cases<<" compression/ring/Engram/RAM/SSD/media/MTP-upgrade/corruption PASS\n";
}
void check_exact_short() {
    CudaExecutionContext execution;
    auto model=make_model(execution);
    auto input=ids(17);
    auto expected=model.hidden_forward(input);
    model.reset(1);
    auto actual=deepseek_v41_prefill_hidden(model,input,17);
    require(actual.sizes()==expected.sizes(),"CED short prompt geometry changed");
    const auto error=(actual.to(tb::kFloat32)-expected.to(tb::kFloat32)).abs().max().item<double>();
    std::cerr<<"CED short maximum error="<<error<<'\n';
    require(std::isfinite(error) && error<.003,"CED short prompt differs from complete forward");
    check_positions(model,17);
    model.hidden_forward(ids(1));check_positions(model,18);
}
void check_short_prefill_tiles(int tile_tokens=8) {
    CudaExecutionContext execution;
    int cases=0;
    for(int ratio:{2,16})for(int context:{127,128,129})for(int tokens:{9,15,32,128}) {
        auto reference=make_model(execution,ratio),candidate=make_model(execution,ratio);
        auto prompt=ids(context),suffix=ids(tokens);
        reference.hidden_forward(prompt);candidate.hidden_forward(prompt);
        std::vector<tb::Tensor> hidden,raw,expected,expected_raw;
        for(int offset=0;offset<tokens;offset+=tile_tokens) {
            auto tile=suffix.narrow(1,offset,std::min(tile_tokens,tokens-offset));
            tb::Tensor wanted_raw,tile_raw;
            expected.push_back(reference.hidden_forward(tile,mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,&wanted_raw));
            expected_raw.push_back(std::move(wanted_raw));
            hidden.push_back(deepseek_v41_prefill_hidden(candidate,tile,tile_tokens,{},&tile_raw));
            raw.push_back(std::move(tile_raw));
        }
        const auto same=[](const tb::Tensor& actual,const tb::Tensor& wanted) {
            require(actual.sizes()==wanted.sizes(),"retained short-prefill geometry differs from ordinary tiles");
            const auto error=(actual.to(tb::kFloat32)-wanted.to(tb::kFloat32)).abs().max().item<double>();
            require(std::isfinite(error) && error<.003,
                "retained short-prefill differs from ordinary causal forward or DSpark history");
        };
        same(tb::cat(hidden,1),tb::cat(expected,1));same(tb::cat(raw,1),tb::cat(expected_raw,1));
        check_positions(candidate,context+tokens);
        same(candidate.hidden_forward(ids(1)),reference.hidden_forward(ids(1)));
        require(candidate.shared->engram_hashes.values==reference.shared->engram_hashes.values,
            "retained short-prefill tiles changed Engram history");
        ++cases;
    }
    std::cout<<"short_prefill_tile_cases="<<cases<<" tile_tokens="<<tile_tokens
        <<" ring/compressor/Engram/DSpark/decode PASS\n";
}
void check_long_and_cancel() {
    CudaExecutionContext execution;
    auto model=make_model(execution);
    tb::Tensor raw;
    auto result=deepseek_v41_prefill_hidden(model,ids(257),64,{},&raw);
    require(raw.sizes()==tb::IntArrayRef({1,128,2*64}),"CED DSpark target tail geometry changed");
    require(result.size(1)==128,"CED did not limit Decoder replay to the local window");
    require(std::isfinite(result.to(tb::kFloat32).abs().max().item<double>()),"CED output is not finite");
    check_positions(model,257);
    model.hidden_forward(ids(1));check_positions(model,258);
    deepseek_v41_prefill_hidden(model,ids(3),2);check_positions(model,261);
    int calls=0;
    auto cancelled=deepseek_v41_prefill_hidden(model,ids(129),32,[&]{return ++calls>=4;});
    require(!cancelled.defined(),"CED cancellation returned partial output");
    check_positions(model,0);
}
void check_retained_prefix() {
    CudaExecutionContext execution;
    int cases=0;
    const auto same=[](const tb::Tensor& a,const tb::Tensor& b,const char* message) {
        const auto error=(a.to(tb::kFloat32)-b.to(tb::kFloat32)).abs().max().item<double>();
        require(std::isfinite(error) && error<.005,message);
    };
    for(bool joint:{false,true})for(int ratio:{2,4})for(int context:{15,16,127,128,129})for(int kept:{0,1,2,3}) {
        auto reference=make_model(execution,ratio),candidate=make_model(execution,ratio);
        const int confirmed=joint?1:0;
        auto prompt=ids(context),draft=ids(3+confirmed);
        reference.hidden_forward(prompt);candidate.hidden_forward(prompt);
        tb::Tensor expected,pending,verified;
        if(joint)pending=reference.hidden_forward(draft.narrow(1,0,1));
        if(kept)expected=reference.hidden_forward(draft.narrow(1,confirmed,kept));
        if(joint) {
            verified=candidate.hidden_forward(draft,mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,nullptr,1);
            same(verified.narrow(1,0,1),pending,"joint verification changed the confirmed token");
        }else {
            candidate.begin_speculative_suffix(3);
            verified=candidate.hidden_forward_speculative_suffix(draft);
        }
        if(kept)same(verified.narrow(1,confirmed,kept),expected,"verified retained prefix differs from causal forward");
        if(joint && kept==3)candidate.commit_speculative();
        else candidate.rollback_speculative(kept);
        check_positions(candidate,context+confirmed+kept);
        for(std::size_t layer=0;layer<candidate.blocks.size();++layer) {
            const auto& actual=dynamic_cast<const deepseek_v41_runtime::Block&>(*candidate.blocks[layer]);
            const auto& wanted=dynamic_cast<const deepseek_v41_runtime::Block&>(*reference.blocks[layer]);
            const int64_t end=context+confirmed+kept,begin=std::max<int64_t>(0,end-128);
            auto slots=tb::arange(begin,end,actual.state.local_kv.options().dtype(tb::kInt64))
                .remainder(128.).to(tb::kInt64);
            same(actual.state.local_kv.index_select(1,slots),wanted.state.local_kv.index_select(1,slots),
                 "retained local KV ring changed accepted rows");
            if(actual.kv_source()) {
                require(actual.state.partial_length==wanted.state.partial_length,"retained compressor count differs");
                const auto count=actual.state.compressed_length;
                same(actual.state.compressed_kv.narrow(1,0,count),wanted.state.compressed_kv.narrow(1,0,count),
                     "retained compressed KV rows changed");
                same(actual.state.index_k.narrow(1,0,count),wanted.state.index_k.narrow(1,0,count),
                     "retained indexer rows changed");
                if(actual.state.partial_length) {
                    same(actual.state.partial_kv.narrow(1,0,actual.state.partial_length),
                         wanted.state.partial_kv.narrow(1,0,wanted.state.partial_length),"retained partial KV differs");
                    same(actual.state.partial_score.narrow(1,0,actual.state.partial_length),
                         wanted.state.partial_score.narrow(1,0,wanted.state.partial_length),"retained partial scores differ");
                }
            }
        }
        same(candidate.hidden_forward(ids(1)),reference.hidden_forward(ids(1)),
             "decode after retained speculative prefix differs");
        require(candidate.shared->engram_hashes.values==reference.shared->engram_hashes.values,
                "retained prefix changed Engram ngram history");
        ++cases;
    }
    std::cout<<"retained_prefix_cases="<<cases<<" suffix/joint compressor/ring/Engram/decode PASS\n";
}
}
int main(int argc,char** argv) try {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    if(argc==2 && std::string(argv[1])=="--modality-routing-check"){check_modality_routing();return 0;}
    if(argc==2 && std::string(argv[1])=="--session-check"){check_session_snapshots();run_dspark_self_check();return 0;}
    if(argc==2 && std::string(argv[1])=="--paged-session-check"){check_paged_sessions();run_dspark_self_check();return 0;}
    if(argc==2 && std::string(argv[1])=="--short-prefill-check"){check_short_prefill_tiles();return 0;}
    if(argc==2 && std::string(argv[1])=="--short-prefill-four-check"){check_short_prefill_tiles(4);return 0;}
    check_modality_routing();check_tail_rope();check_exact_short();check_short_prefill_tiles();check_short_prefill_tiles(4);check_long_and_cancel();check_retained_prefix();check_session_snapshots();check_paged_sessions();run_dspark_self_check();
    std::cout<<"DeepSeek-V4.1 CED numerical/cache tests passed\n";return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
