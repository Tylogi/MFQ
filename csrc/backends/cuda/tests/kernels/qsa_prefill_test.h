#pragma once
#include "native/env.h"
#include <filesystem>

void qsa_scalar_history_chain_case() {
    int cases=0,scalar_replays=0,chain_replays=0;
    for(int mode:{0,1,2})for(int width:{32,256}) {
        CudaExecutionContext execution;
        constexpr int hidden=32,heads=4,kv=2,iw=8,ih=2,pool=4,maximum=128,budget=16;
        auto rotary=std::make_shared<RotaryEmbedding>(8,maximum,10000.);
        auto norm=data({width},1,0),inorm=data({iw},1,0);
        QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
            linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,inorm,inorm};
        QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
        std::shared_ptr<KvMemoryManager> memory;
        if(mode) {
            KvMemoryOptions policy;policy.placement=mode==1?"auto":"auto_ssd";policy.budget_policy=true;
            policy.gpu_budget=2*(maximum+maximum/pool)*iw*2+8*kv*width*4;
            policy.ram_limit_set=true;policy.ram_limit=mode==1?2*maximum*kv*width*4:256;
            policy.directory=(std::filesystem::current_path()/"qsa-scalar-history-chain-test").string();
            memory=std::make_shared<KvMemoryManager>(policy);
        }
        Qsa actual(weights,cfg,rotary,true,true,{},memory),oracle(weights,cfg,rotary,true,true,{},memory);
        if(memory)memory->finalize();
        auto inputs=data({1,maximum,hidden},17,.4f);
        auto history=tb::arange(6,inputs.options().dtype(tb::kInt64));
        actual.forward(execution,inputs.narrow(1,0,6),history,history,true);
        oracle.forward(execution,inputs.narrow(1,0,6),history,history,true);
        actual.prepare_graph(history,maximum);oracle.prepare_graph(history,maximum);
        std::map<int,std::unique_ptr<Cell>> chains;
        Cell scalar(1,hidden),next_actual(1,hidden),next_oracle(1,hidden);
        const auto compare_state=[&] {
            if(!actual.supports_session())return;
            Qwen4BlockSessionState a,b;std::size_t bytes=0;
            actual.capture_session(a,actual.position(),bytes);oracle.capture_session(b,oracle.position(),bytes);
            equal(a.keys,b.keys,"scalar-chain KV keys",0,0);
            equal(a.values,b.values,"scalar-chain KV values",0,0);
            equal(a.index,b.index,"scalar-chain indexer",0,0);
        };
        const auto run=[&](int tokens) {
            const auto start=actual.position();
            auto positions=tb::arange(start,start+tokens,history.options());
            auto& cell=chains[tokens];if(!cell)cell=std::make_unique<Cell>(tokens,hidden);
            cell->x.copy_(inputs.narrow(1,start,tokens));cell->pos.copy_(positions);
            if(!cell->window.valid())cell->window.capture([&] {
                for(int row=0;row<tokens;++row)
                    actual.forward_graph(execution,cell->x.narrow(1,row,1),cell->pos.narrow(0,row,1),
                        cell->pos.narrow(0,row,1),{},true,true);
            },[]{},[]{});
            cell->window.run();actual.advance_graph(tokens);++chain_replays;
            for(int row=0;row<tokens;++row) {
                scalar.x.copy_(inputs.narrow(1,start+row,1));scalar.pos.copy_(positions.narrow(0,row,1));
                if(!scalar.window.valid())scalar.window.capture([&] {
                    oracle.forward_graph(execution,scalar.x,scalar.pos,scalar.pos,{},true,true);
                },[]{},[]{});
                scalar.window.run();oracle.advance_graph(1);++scalar_replays;
            }
            history=tb::cat({history,positions},-1);compare_state();
            const auto next=actual.position();
            auto next_positions=tb::arange(next,next+1,history.options());
            for(auto* next_cell:{&next_actual,&next_oracle}) {
                next_cell->x.copy_(inputs.narrow(1,next,1));next_cell->pos.copy_(next_positions);
            }
            if(!next_actual.window.valid())next_actual.window.capture([&] {
                next_actual.output=actual.forward_graph(execution,next_actual.x,next_actual.pos,next_actual.pos);
            },[]{},[&]{next_actual.output={};});
            if(!next_oracle.window.valid())next_oracle.window.capture([&] {
                next_oracle.output=oracle.forward_graph(execution,next_oracle.x,next_oracle.pos,next_oracle.pos);
            },[]{},[&]{next_oracle.output={};});
            next_actual.window.run();actual.advance_graph(1);
            next_oracle.window.run();oracle.advance_graph(1);
            equal(next_actual.output,next_oracle.output,"scalar-chain next-token readout",0,0);
            history=tb::cat({history,next_positions},-1);compare_state();++cases;
        };
        for(int tokens:{2,3,4,6,8,2,3})run(tokens);
        actual.truncate(13);oracle.truncate(13);history=history.narrow(0,0,13);
        run(4);run(2);
    }
    std::cout<<"QSA scalar history chain exact KV/index/readout, repeated graph, rollback, GPU/RAM/SSD PASS cases="
        <<cases<<" scalar_replays="<<scalar_replays<<" chain_replays="<<chain_replays<<'\n';
}

void qsa_history_prime_case() {
    int cases=0;
    for(int mode:{0,1,2}) {
        CudaExecutionContext execution;
        constexpr int hidden=32,heads=4,kv=2,width=32,iw=8,ih=2,pool=4,maximum=512,budget=64;
        auto rotary=std::make_shared<RotaryEmbedding>(8,maximum,10000.);
        auto norm=data({width},1,0),inorm=data({iw},1,0);
        QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
            linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,inorm,inorm};
        QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
        std::shared_ptr<KvMemoryManager> memory;
        if(mode) {
            KvMemoryOptions policy;policy.placement=mode==1?"auto":"auto_ssd";policy.budget_policy=true;
            policy.gpu_budget=(maximum+maximum/pool)*iw*2+(mode==1?maximum:4)*kv*width*4;
            policy.ram_limit_set=true;policy.ram_limit=mode==1?maximum*kv*width*4:256;
            policy.directory=(std::filesystem::current_path()/"qsa-history-prime-test").string();
            memory=std::make_shared<KvMemoryManager>(policy);
        }
        Qsa actual(weights,cfg,rotary,true,true,{},memory),oracle(weights,cfg,rotary);
        if(memory)memory->finalize();
        auto inputs=data({1,maximum,hidden},17,.4f);
        int64_t offset=0;
        for(int tokens:{17,128,257}) {
            auto positions=tb::arange(offset,offset+tokens,inputs.options().dtype(tb::kInt64));
            auto full=tb::arange(offset+tokens,positions.options());
            auto x=inputs.narrow(1,offset,tokens);
            oracle.forward(execution,x,positions,full,true);
            if(actual.forward(execution,x,positions,full,true,nullptr,{},true,true).defined())
                throw std::runtime_error("history priming computed an unused attention output");
            offset+=tokens;
            if(actual.position()!=offset)throw std::runtime_error("history priming did not advance KV");
            if(actual.supports_session()) {
                Qwen4BlockSessionState a,b;std::size_t bytes=0;
                actual.capture_session(a,offset,bytes);oracle.capture_session(b,offset,bytes);
                equal(a.keys,b.keys,"history priming keys",0,0);
                equal(a.values,b.values,"history priming values",0,0);
                equal(a.index,b.index,"history priming index",0,0);
            }
            auto next=tb::arange(offset,offset+1,positions.options());
            actual.prepare_graph(full,maximum);
            {
                auto next_x=inputs.narrow(1,offset,1);
                DecodeWindow window(mfq_current_cuda_stream());
                window.capture([&] {actual.forward_graph(execution,next_x,next,next,{},true,true);},[]{},[]{});
                window.run();actual.advance_graph(1);
            }
            oracle.forward(execution,inputs.narrow(1,offset,1),next,tb::arange(offset+1,positions.options()),true);
            actual.leave_graph();++offset;
            next=tb::arange(offset,offset+1,positions.options());
            full=tb::arange(offset+1,positions.options());x=inputs.narrow(1,offset,1);
            equal(actual.forward(execution,x,next,full,true),oracle.forward(execution,x,next,full,true),
                "history priming next-token readout");
            ++offset;++cases;
        }
        actual.truncate(65);oracle.truncate(65);
        auto positions=tb::arange(65,73,inputs.options().dtype(tb::kInt64));
        auto full=tb::arange(73,positions.options()),x=inputs.narrow(1,65,8);
        equal(actual.forward(execution,x,positions,full,true),oracle.forward(execution,x,positions,full,true),
            "history priming rollback and continuation");++cases;
    }
    std::cout<<"QSA cache-only priming cases="<<cases<<" exact KV/index and continuation oracle PASS\n";
}

void qsa_prefill_fusion_case() {
    struct Setting {
        std::string old;
        Setting(){if(const auto* p=mfq::cuda::env_lookup("MFQ_QSA_PREFILL_FUSED"))old=p;}
        static void set(const char* p) {
            if(*p)mfq::cuda::env_set("MFQ_QSA_PREFILL_FUSED",p);else mfq::cuda::env_clear("MFQ_QSA_PREFILL_FUSED");
        }
        ~Setting(){set(old.c_str());}
    } setting;
    int cases=0,grouped_calls=0;
    for(int width:{8,256})for(bool grouped:{false,true}) {
        CudaExecutionContext execution;
        constexpr int hidden=32,kv=2,iw=8,ih=2,pool=4,maximum=2048,budget=512;
        const int heads=width==256?24:4;
        auto rotary=std::make_shared<RotaryEmbedding>(std::min(width,8),maximum,10000.);
        auto norm=data({width},1,0),inorm=data({iw},1,0);
        QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
            linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,inorm,inorm};
        if(grouped)weights.input_projection=[q=weights.query,k=weights.key,v=weights.value,i=weights.index_query_key,&grouped_calls]
                (CudaExecutionContext& context,const Tensor& x) {
            ++grouped_calls;return std::vector<Tensor>{q(context,x),k(context,x),v(context,x),i(context,x)};
        };
        QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
        Qsa actual(weights,cfg,rotary),oracle(weights,cfg,rotary);
        auto inputs=data({1,maximum,hidden},17,.4f);
        int64_t offset=0;
        for(int tokens:{128,257,511,512,1}) {
            auto positions=tb::arange(offset,offset+tokens,inputs.options().dtype(tb::kInt64));
            auto history=tb::arange(offset+tokens,positions.options());
            auto x=inputs.narrow(1,offset,tokens);
            Setting::set("0");auto expected=oracle.forward(execution,x,positions,history,true);
            Setting::set("1");auto result=actual.forward(execution,x,positions,history,true);
            equal(result,expected,"QSA prefill fused projection/normalization/RoPE");
            ++cases;offset+=tokens;
        }
        actual.truncate(385);oracle.truncate(385);
        auto positions=tb::arange(385,642,inputs.options().dtype(tb::kInt64));
        auto history=tb::arange(642,positions.options());auto x=inputs.narrow(1,385,257);
        Setting::set("0");auto expected=oracle.forward(execution,x,positions,history,true);
        Setting::set("1");auto result=actual.forward(execution,x,positions,history,true);
        equal(result,expected,"QSA prefill cache truncation and replay");++cases;
    }
    if(grouped_calls!=10)throw std::runtime_error("QSA prefill grouped projections were not used for every large chunk");
    std::cout<<"QSA prefill fusion cases="<<cases<<" grouped_calls="<<grouped_calls
             <<" dense/sparse prefixes, changing chunk sizes, decode transition, cache truncation PASS\n";
}
