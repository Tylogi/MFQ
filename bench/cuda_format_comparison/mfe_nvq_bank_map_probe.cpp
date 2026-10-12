// Reuse the real-model FFN fixture and its independent unfused oracle.
#define main bank_fixture_main
#include "mfq_mfe_bench.cpp"
#undef main

static void bind_compact_production(Call& call,int group_hint=0) {
    auto projections=call.layer.projections;
    for(auto& projection:projections)for(auto& pool:projection.pools) {
        if(pool.family!=MixedMoeFamily::Nvq)continue;
        auto& w=pool.nvq;
        const auto found=call.layer.dense_e8_streams.find(w.indices_packed.data_ptr());
        if(found==call.layer.dense_e8_streams.end())continue;
        w.indices_packed=found->second;
        w.aux_packed=tb::empty({0},w.indices_packed.options());
        w.sub_scale_packed=tb::empty({0},w.indices_packed.options());
        w.dense_groups=true;
    }
    std::array<const NintWeight*,3> shared{};
    if(call.shared_gate.defined())for(int p=0;p<3;++p)shared[p]=&call.layer.shared[p].nint;
    call.fused=std::make_unique<MfeFfnRuntime>(
        std::array<MixedMoeRuntime*,3>{&projections[0],&projections[1],&projections[2]},
        call.x,call.ids,call.weights,call.table,shared,call.shared_gate,0,0,nullptr,false,false,group_hint);
    call.fused->run();
}

static void probe_exact(const tb::Tensor& expected,const tb::Tensor& actual) {
    if(expected.sizes()!=actual.sizes() || expected.scalar_type()!=actual.scalar_type() ||
       std::memcmp(expected.data_ptr(),actual.data_ptr(),expected.numel()*expected.element_size()))
        throw std::runtime_error("cached map changed FFN output bits");
}

static void bank_map_option(bool enabled) {
#ifdef _WIN32
    if(_putenv_s("MFQ_MFE_NVQ_BANK_MAP",enabled?"1":"0"))throw std::runtime_error("bank map option");
#else
    if(setenv("MFQ_MFE_NVQ_BANK_MAP",enabled?"1":"0",1))throw std::runtime_error("bank map option");
#endif
}

static std::unique_ptr<Call> bank_call(Layer& layer,bool shared,int tokens,bool enabled,int group_hint=0) {
    bank_map_option(enabled);
    auto call=std::make_unique<Call>(layer,shared,tokens);
    bind_compact_production(*call,group_hint);
    return call;
}

int main(int argc,char** argv)try {
    if(argc!=3)throw std::runtime_error("expected model-first-shard and benchmark mode");
    auto context=mfq::cuda::default_context(0);
    auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    auto model=mfq::open_model_source(argv[1]);
    std::vector<std::unique_ptr<Layer>> layers;
    for(int layer:{0,8,16,24,32,40})layers.push_back(std::make_unique<Layer>(*model,layer));
    const auto mode=std::string(argv[2]);
    if(mode=="--capacity-sweep") {
        constexpr int capacities[]{2,4,8};
        for(bool shared:{false,true}) {
            std::array<std::vector<std::unique_ptr<Call>>,3> variants;
            for(int candidate=0;candidate<3;++candidate)for(auto& layer:layers) {
#ifdef _WIN32
                if(_putenv_s("MFQ_MFE_TOKEN_GROUP_CAPACITY",""))throw std::runtime_error("group capacity option");
#else
                if(unsetenv("MFQ_MFE_TOKEN_GROUP_CAPACITY"))throw std::runtime_error("group capacity option");
#endif
                auto call=bank_call(*layer,shared,8,true,capacities[candidate]==2?0:capacities[candidate]);
                if(call->fused->batch().token_group_capacity!=capacities[candidate])
                    throw std::runtime_error("sweep hint did not select requested capacity");
                variants[candidate].push_back(std::move(call));
            }
            for(size_t layer=0;layer<layers.size();++layer)for(int candidate=1;candidate<3;++candidate) {
                probe_exact(variants[0][layer]->fused->output().cpu(),variants[candidate][layer]->fused->output().cpu());
            }
            std::array<MfqCudaGraph,3> graphs;
            mfq_prepare_cuda_graph_memory(graphs[0]);
            for(int candidate=0;candidate<3;++candidate) {
                for(auto& call:variants[candidate])call->fused->run();
                if(candidate)graphs[candidate].capture_begin_shared();else graphs[candidate].capture_begin();
                for(auto& call:variants[candidate])call->fused->run();
                graphs[candidate].capture_end();
            }
            std::array<std::vector<double>,3> times;
            for(int repeat=0;repeat<6;++repeat)for(int offset=0;offset<3;++offset) {
                const int candidate=repeat<3?(repeat+offset)%3:(repeat+3-offset)%3;
                times[candidate].push_back(fb::median(fb::measure([&]{graphs[candidate].replay();},
                    stream.stream(),layers.size())));
            }
            for(int candidate=0;candidate<3;++candidate)
                std::cout<<"actual_mfe_capacity_sweep tokens=8 shared="<<shared
                    <<" capacity="<<capacities[candidate]<<" us="<<fb::median(times[candidate])<<"\n"<<std::flush;
        }
        std::cout<<"actual_mfe_capacity_sweep PASS\n";
        return 0;
    }
    const bool grouped=mode=="--grouped-check" || mode=="--grouped-bench";
    const bool capacity4=mode=="--capacity4-check" || mode=="--capacity4-bench";
    const bool capacity=capacity4 || mode=="--capacity-check" || mode=="--capacity-bench";
    const int group_hint=capacity4?4:8;
    const std::string label=grouped?"actual_mfe_grouped_bank_map":
        capacity?"actual_mfe_capacity2_vs"+std::to_string(group_hint):"actual_mfe_bank_map";
    const bool verify=mode=="--check" || mode=="--capacity-check" || mode=="--capacity4-check" || mode=="--grouped-check";
    if(!verify && mode!="--bench" && !capacity && !grouped)throw std::runtime_error("unknown benchmark mode");
    const auto group_option=[](const char* value) {
#ifdef _WIN32
        if(_putenv_s("MFQ_MFE_TOKEN_GROUP_CAPACITY",value))throw std::runtime_error("group capacity option");
#else
        if(setenv("MFQ_MFE_TOKEN_GROUP_CAPACITY",value,1))throw std::runtime_error("group capacity option");
#endif
    };
    int map_views=0,checks=0;
    for(int tokens:capacity || grouped?std::vector<int>{8}:
        verify?std::vector<int>{1,2,3,4,5,6,8}:std::vector<int>{1,3,6,8})
    for(bool shared:{false,true}) {
        std::vector<std::unique_ptr<Call>> baseline,cached;
        for(auto& layer:layers) {
            if(capacity || grouped)group_option("");
            baseline.push_back(bank_call(*layer,shared,tokens,capacity,grouped?8:0));
            cached.push_back(bank_call(*layer,shared,tokens,true,capacity?group_hint:grouped?8:0));
            const auto& batch=cached.back()->fused->batch();
            if(capacity && (baseline.back()->fused->batch().token_group_capacity!=2 ||
                batch.token_group_capacity!=group_hint))throw std::runtime_error("group hint did not select actual capacities");
            if(grouped && (baseline.back()->fused->batch().token_group_capacity!=8 ||
                batch.token_group_capacity!=8))throw std::runtime_error("grouped mapping test did not select capacity8");
            std::vector<mfq::cuda::MfePackedProjection> views(batch.projection_counts[0]+
                batch.projection_counts[1]+batch.projection_counts[2]);
            fb::check(cudaMemcpyAsync(views.data(),batch.projections,views.size()*sizeof(views[0]),
                cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
            fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            for(const auto& d:views)if(d.e8_dense_groups && (d.format==5 || d.format==13 || d.format==14)) {
                std::array<int8_t,64> metadata{};
                fb::check(cudaMemcpy(metadata.data(),d.fields[4],metadata.size(),cudaMemcpyDeviceToHost));
                uint32_t expected=0;
                for(unsigned state=0;state<16;++state)expected|=uint32_t(uint8_t(metadata[36+state]))<<(2*state);
                if(d.nvq_bank_map!=expected)throw std::runtime_error("actual compact descriptor lost bank map");
                map_views+=expected!=0;
            }
            if(verify) {
                auto& old=*baseline.back();auto& current=*cached.back();
                MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
                graph.capture_begin();current.fused->run();graph.capture_end();
                for(int step=0;step<3;++step) {
                    std::vector<int32_t> selected(tokens*routes);
                    for(int i=0;i<tokens*routes;++i) {
                        selected[i]=layer->selected[(i/(step==1?2:1)+step)%routes];
                        if(step==2 && i%3==0)selected[i]=-1;
                        if(step==2 && i%3==1)selected[i]=experts;
                    }
                    const auto ids=tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA);
                    old.ids.copy_(ids);current.ids.copy_(ids);
                    auto values=fb::input(tokens,hidden);
                    for(size_t i=0;i<values.size();++i)values[i]=step==0?0.f:
                        values[i]*float(step)+std::sin(float(i)*.019f+step)*.21f;
                    const auto input=tb::tensor(values).reshape({tokens,hidden}).to(tb::kCUDA,tb::kFloat16);
                    old.x.copy_(input);current.x.copy_(input);
                    old.fused->run();graph.replay();
                    probe_exact(old.fused->output().cpu(),current.fused->output().cpu());
                    auto a=old.fused->hidden().cpu(),b=current.fused->hidden().cpu();
                    for(int token=0;token<tokens;++token)for(int route=0;route<routes+int(shared);++route) {
                        if(route<routes && unsigned(selected[token*routes+route])>=unsigned(experts))continue;
                        const auto slot=token*(routes+int(shared))+route;
                        const auto offset=size_t(slot)*batch.hidden_stride*a.element_size();
                        const auto width=route==routes?batch.shared_intermediate:batch.intermediate;
                        if(std::memcmp(static_cast<const char*>(a.data_ptr())+offset,
                                       static_cast<const char*>(b.data_ptr())+offset,width*a.element_size()))
                            throw std::runtime_error("cached map changed active hidden bits");
                    }
                    ++checks;
                }
            }
        }
        if(!verify) {
            MfqCudaGraph old_graph,new_graph;mfq_prepare_cuda_graph_memory(old_graph);
            for(auto& call:baseline)call->fused->run();
            old_graph.capture_begin();for(auto& call:baseline)call->fused->run();old_graph.capture_end();
            for(auto& call:cached)call->fused->run();
            new_graph.capture_begin_shared();for(auto& call:cached)call->fused->run();new_graph.capture_end();
            std::vector<double> old_times,new_times;
            for(int repeat=0;repeat<3;++repeat) {
                const auto measure_old=[&]{old_times.push_back(fb::median(fb::measure([&]{old_graph.replay();},
                    stream.stream(),layers.size())));};
                const auto measure_new=[&]{new_times.push_back(fb::median(fb::measure([&]{new_graph.replay();},
                    stream.stream(),layers.size())));};
                if(repeat&1){measure_new();measure_old();}else{measure_old();measure_new();}
            }
            std::cout<<label<<" tokens="<<tokens<<" shared="<<shared
                <<" old_us="<<fb::median(old_times)<<" cached_us="<<fb::median(new_times)<<"\n"<<std::flush;
        }
    }
    if(!map_views)throw std::runtime_error("no compact FFN actually selected cached bank mapping");
    std::cout<<label<<" checks="<<checks<<" mapped_views="<<map_views<<" PASS\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
