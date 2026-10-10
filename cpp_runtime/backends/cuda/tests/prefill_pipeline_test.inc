void verify_prefill_graph_execution(const std::filesystem::path& root,bool grouped=false,int group_size=1,bool sparse=false,bool oracle=false,bool layer_prefill=false,bool layout_only=false,bool batch_check=false) {
    struct Setting {
        std::string old;
        std::string old_direct,old_groups,old_group_size,old_mma,old_layer,old_retain;
        Setting(){if(const auto* p=std::getenv("MFQ_MOE_PREFILL_GRAPH"))old=p;
            if(const auto* p=std::getenv("MFQ_MOE_PREFILL_DIRECT_RAM"))old_direct=p;
            if(const auto* p=std::getenv("MFQ_MOE_PREFILL_GROUPS"))old_groups=p;
            if(const auto* p=std::getenv("MFQ_MOE_PREFILL_GROUP_SIZE"))old_group_size=p;
            if(const auto* p=std::getenv("MFQ_MOE_PREFILL_COMPACT_MMA"))old_mma=p;
            if(const auto* p=std::getenv("MFQ_MOE_PREFILL_LAYER"))old_layer=p;
            if(const auto* p=std::getenv("MFQ_MOE_PREFILL_POOL_RETAIN"))old_retain=p;}
        static void set_env(const char* key,const char* p) {
#ifdef _WIN32
            _putenv_s(key,p);
#else
            if(*p)setenv(key,p,1);else unsetenv(key);
#endif
        }
        static void set(const char* p){set_env("MFQ_MOE_PREFILL_GRAPH",p);}
        ~Setting(){set(old.c_str());set_env("MFQ_MOE_PREFILL_DIRECT_RAM",old_direct.c_str());set_env("MFQ_MOE_PREFILL_GROUPS",old_groups.c_str());set_env("MFQ_MOE_PREFILL_GROUP_SIZE",old_group_size.c_str());set_env("MFQ_MOE_PREFILL_COMPACT_MMA",old_mma.c_str());set_env("MFQ_MOE_PREFILL_LAYER",old_layer.c_str());set_env("MFQ_MOE_PREFILL_POOL_RETAIN",old_retain.c_str());}
    } setting;
    int cases=0,direct_checks=0,pool_checks=0,reuse_checks=0,batch_checks=0,wide_row_bits_checks=0;double maximum_relative=0,maximum_batch_relative=0;
    for(const auto& names:std::array<std::array<std::string,3>,3>{
        std::array<std::string,3>{"nint5","nint5","nint5"},
        {"nvq3j-512","nvq3j-512","nvq3j-512"},
        {"nint6","nvq3j-l","nvq2j-xl"}}) {
        std::array<std::shared_ptr<const mfq::ModelSource>,3> canonical;
        if(oracle)for(int p=0;p<3;++p)
            canonical[p]=mfq::open_model_source((root/(names[p]+(p==2?"-2560-640.mfq":"-640-2560.mfq"))).string());
        std::array<CudaExecutionContext,2> execution;
        std::array<std::shared_ptr<MoeExpertCache>,2> caches;
        std::array<std::array<MoeFfnForward,2>,2> forward;
        const std::array<MfqCudaStream,2> streams={mfq_get_stream_from_pool(),mfq_get_current_cuda_stream()};
        for(int mode=0;mode<2;++mode) {
            Setting::set_env("MFQ_MOE_PREFILL_LAYER",layer_prefill && mode==1?"1":"0");
            MfqCudaStreamGuard guard(streams[mode]);
            auto& cfg=execution[mode].config;
            execution[mode].force_moe_prefill_mma_off=sparse && !oracle;
            cfg.moe_pipeline=true;cfg.moe_preload_all=true;cfg.moe_assert_resident=true;
            cfg.moe_hybrid_cpu=false;cfg.moe_direct_ram=true;cfg.moe_ram_pcie=true;
            cfg.moe_ram_pcie_fraction=1.;
            // Exercise prefill/decode/repeated-prefill with adaptation requested.
            // The immutable field-major layout must preserve expert ownership.
            cfg.moe_residency_adapt=layer_prefill && mode==1;
            std::array<std::shared_ptr<MoeQuantRangeSource>,3> sources;
            std::vector<std::shared_ptr<MoeQuantRangeSource>> registered;
            int64_t bundle=0;
            for(int p=0;p<3;++p) {
                sources[p]=source(root/(names[p]+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                bundle+=bytes(sources[p]->metadata()->pools[0]);
            }
            cfg.moe_host_cache_bytes=12*bundle;caches[mode]=make_moe_expert_cache(2*bundle,cfg);
            for(int layer=0;layer<2;++layer) {
            std::vector<std::shared_ptr<MfeWeight>> gu,down;
            // Qwen4 registers Down first, with an architecture role string.
            // Bundle order, rather than registration order or role, identifies it.
            for(int p:{2,0,1}) {
                auto layer_source=source(root/(names[p]+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                registered.push_back(layer_source);
                auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(caches[mode],
                    "prefill/"+std::to_string(layer)+"/"+std::to_string(p),layer_source,1,layer,
                    p==2?"qwen4_exp":p==0?"gate":"up"));
                (p==2?down:gu).push_back(weight);
            }
            forward[mode][layer]=make_moe_ffn_pipeline(gu,down);
            }
            finalize_moe_expert_cache(caches[mode]);
            if(layer_prefill && mode==1)for(const auto& quant:registered)
                for(std::size_t pool=0;pool<quant->store().pool_count();++pool)
                    quant->store().visit_pool_experts(pool,[&](int expert,mfq::MfeQuantExpert encoded) {
                        if(quant->gpu_resident(expert))return;
                        auto actual=quant->acquire_expert(expert);
                        const auto expected=moe_cache_fields(quant->decode_expert(expert,encoded));
                        const auto fields=moe_cache_fields(actual->weights);
                        for(std::size_t f=0;f<fields.size();++f)
                            if(fields[f].sizes()!=expected[f].sizes() || std::memcmp(fields[f].data_ptr(),expected[f].data_ptr(),tensor_nbytes(fields[f])))
                                throw std::runtime_error("layer RAM field bytes differ: expert="+std::to_string(expert)+" field="+std::to_string(f));
                    });
        }
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        cudaMemPool_t pool=nullptr;
        uint64_t initial_threshold=0;
        int pool_supported=0;
        MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&pool_supported,cudaDevAttrMemoryPoolsSupported,mfq_current_cuda_device()));
        if(pool_supported) {
            MFQ_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,mfq_current_cuda_device()));
            MFQ_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&initial_threshold));
        }
#endif
        const std::vector<int> token_cases=batch_check?
            std::vector<int>{1,16,31,64,512,513,1,16}:
            std::vector<int>{1,16,31,64,512,1,16};
        for(int tokens:token_cases)for(int step=0;step<3;++step) {
            std::vector<float> values(tokens*2560),weights(tokens*3);
            std::vector<int32_t> ids(tokens*3);
            for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*37)*.017f)*.11f;
            for(size_t i=0;i<ids.size();++i){ids[i]=(int(i)+step)%3;weights[i]=.13f+float(i%7)*.01f;}
            if(step==2)std::fill(ids.begin(),ids.end(),2);
            if(sparse && tokens>8 && step==1) {
                std::fill(ids.begin(),ids.end(),0);
                for(int i=0;i<3;++i)ids[i]=1;
                for(int i=3;i<8;++i)ids[i]=2;
            }
            auto x=tb::tensor(values).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA);
            auto route=tb::tensor(ids).reshape({tokens,3}).to(tb::kCUDA);
            auto w=tb::tensor(weights).reshape({tokens,3}).to(tb::kCUDA);
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            std::array<std::array<tb::Tensor,2>,2> queued;
            for(int mode=0;mode<2;++mode) {
                MfqCudaStreamGuard guard(streams[mode]);
                Setting::set(mode==0?"1":"0");
                Setting::set_env("MFQ_MOE_PREFILL_COMPACT_MMA",oracle && mode==1?"1":"0");
                Setting::set_env("MFQ_MOE_PREFILL_DIRECT_RAM",mode==0?"0":"1");
                Setting::set_env("MFQ_MOE_PREFILL_GROUPS",grouped && mode==1?"1":"0");
                Setting::set_env("MFQ_MOE_PREFILL_GROUP_SIZE",std::to_string(group_size).c_str());
                Setting::set_env("MFQ_MOE_PREFILL_LAYER",layer_prefill && !layout_only && mode==1 && tokens>8?"1":"0");
                Setting::set_env("MFQ_MOE_PREFILL_POOL_RETAIN",mode==1?"1":"0");
                for(int layer=0;layer<2;++layer) {
                    queued[mode][layer]=forward[mode][layer](execution[mode],x,route,w);
                    if(layer_prefill && !layout_only && mode==1 && tokens>8) {
                        queued[mode][layer]=forward[mode][layer](execution[mode],x,route,w);
                        ++reuse_checks;
                    }
                }
#ifdef MFQ_NATIVE_CUDA_RUNTIME
                if(mode==1 && pool) {
                    uint64_t threshold=0;
                    MFQ_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&threshold));
                    const auto limit=mfq::cuda::default_context(mfq_current_cuda_device())->memory_stats().limit;
                    const uint64_t expected=tokens>8?(limit?limit:std::numeric_limits<uint64_t>::max()):initial_threshold;
                    if(threshold!=expected)throw std::runtime_error("prefill pool retention/decode restoration differs");
                    ++pool_checks;
                }
#endif
            }
            tb::Tensor dense_reference;
            if(batch_check && tokens>=512) {
                MfqCudaStreamGuard guard(streams[1]);
                for(int layer=0;layer<2;++layer) {
                    // A one-row final chunk uses Q8 decode arithmetic. Compare
                    // full batches directly to establish the new tile's bits.
                    const auto* wide_option=std::getenv("MFQ_NVQ_PREFILL_M256");
                    const auto* k_option=std::getenv("MFQ_NVQ_PREFILL_NARROW_G2");
                    const auto* cohorts_option=std::getenv("MFQ_NVQ_PREFILL_COHORTS");
                    const bool wide=wide_option && std::atoi(wide_option)!=0;
                    const bool k_tile=k_option && std::atoi(k_option)!=0;
                    const char* tile_key=wide?"MFQ_NVQ_PREFILL_M256":k_tile?"MFQ_NVQ_PREFILL_NARROW_G2":"MFQ_NVQ_PREFILL_COHORTS";
                    if(wide || k_tile || cohorts_option && std::atoi(cohorts_option)!=0) {
                        const std::string saved_wide=wide?wide_option:k_tile?k_option:cohorts_option;
                        auto actual=queued[1][layer].cpu().contiguous();
                        Setting::set_env(tile_key,"0");
                        auto original=forward[1][layer](execution[1],x,route,w).cpu().contiguous();
                        Setting::set_env(tile_key,saved_wide.c_str());
                        if(actual.sizes()!=original.sizes() ||
                           std::memcmp(actual.data_ptr(),original.data_ptr(),actual.nbytes()))
                            throw std::runtime_error("NVQ tile variant changed original FFN output bits");
                        ++wide_row_bits_checks;
                    }
                    std::vector<tb::Tensor> parts;
                    for(int start=0;start<tokens;start+=64) {
                        const int length=std::min(64,tokens-start);
                        parts.push_back(forward[1][layer](execution[1],x.narrow(0,start,length).contiguous(),
                            route.narrow(0,start,length).contiguous(),w.narrow(0,start,length).contiguous()));
                    }
                    auto expected=tb::cat(parts,0).cpu().to(tb::kFloat32);
                    auto actual=queued[1][layer].cpu().to(tb::kFloat32);
                    double error=0,scale=0;
                    for(int64_t i=0;i<actual.numel();++i) {
                        const double delta=double(actual.data_ptr<float>()[i])-expected.data_ptr<float>()[i];
                        error+=delta*delta;scale+=double(expected.data_ptr<float>()[i])*expected.data_ptr<float>()[i];
                    }
                    const double rms=std::sqrt(error/std::max(1e-30,scale));
                    maximum_batch_relative=std::max(maximum_batch_relative,rms);++batch_checks;
                    std::cout<<"prefill_expert_batch_oracle format="<<names[0]<<" rows="<<tokens<<" step="<<step
                        <<" layer="<<layer<<" relative_rms="<<rms<<std::endl;
                    if(!std::isfinite(rms) || rms>.002)throw std::runtime_error("FFN cross-batch numerical mismatch");
                }
            }
            if(oracle && tokens>8) {
                MfqCudaStreamGuard guard(streams[1]);
                auto gate=mfe_dense_reference(*canonical[0],"linear.weight",x,ids,tokens,3,false).reshape({tokens,3,640});
                auto up=mfe_dense_reference(*canonical[1],"linear.weight",x,ids,tokens,3,false).reshape({tokens,3,640});
                auto hidden=moe_swiglu_rounded_cuda(gate,up,moe_swiglu_sigmoid_table_cuda());
                dense_reference=mfe_dense_reference(*canonical[2],"linear.weight",hidden,ids,tokens,3,true)
                    .reshape({tokens,3,2560}).cpu().to(tb::kFloat32);
            }
            for(int layer=0;layer<2;++layer) {
            std::array<tb::Tensor,2> result;
            for(int mode=0;mode<2;++mode) {
                MfqCudaStreamGuard guard(streams[mode]);
                result[mode]=queued[mode][layer].cpu().contiguous();
                if(layer_prefill && mode==1 && tokens==1)MoeResidencyTestAccess::verify_transfer_fields(*caches[mode]);
            }
            if(result[0].sizes()!=result[1].sizes())throw std::runtime_error("prefill FFN output shape differs");
            if(oracle && tokens>8) {
                auto actual=result[1].to(tb::kFloat32);double squared=0,reference_squared=0;
                float difference=0,scale=0;
                for(int64_t i=0;i<actual.numel();++i) {
                    const float a=actual.data_ptr<float>()[i],b=dense_reference.data_ptr<float>()[i];
                    if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite streamed FFN oracle output");
                    const double d=double(a)-b;squared+=d*d;reference_squared+=double(b)*b;
                    difference=std::max(difference,float(std::abs(d)));scale=std::max(scale,std::abs(b));
                }
                const double relative=std::sqrt(squared/std::max(reference_squared,1e-20));
                maximum_relative=std::max(maximum_relative,relative);
                // Rare NINT groups can use Q8 activations; NVQ prefill is FP16.
                const double tolerance=names[0].starts_with("nint")?.02:.002;
                if(relative>tolerance || difference>.02+1.5*tolerance*scale)
                    throw std::runtime_error("streamed FFN differs from canonical weights: format="+names[0]+" rows="+
                        std::to_string(tokens)+" step="+std::to_string(step)+" relative_rms="+std::to_string(relative));
            } else if(std::memcmp(result[0].data_ptr(),result[1].data_ptr(),result[0].numel()*result[0].element_size())) {
                auto a=result[0].to(tb::kFloat32),b=result[1].to(tb::kFloat32);
                float difference=0,scale=0;int64_t worst=0;
                for(int64_t i=0;i<a.numel();++i) {
                    const float d=std::abs(a.data_ptr<float>()[i]-b.data_ptr<float>()[i]);
                    if(d>difference){difference=d;worst=i;}scale=std::max(scale,std::abs(a.data_ptr<float>()[i]));
                }
                std::cerr<<"prefill_difference format="<<names[0]<<" tokens="<<tokens<<" step="<<step
                         <<" layer="<<layer<<" max_abs="<<difference<<" scale="<<scale<<" index="<<worst
                         <<" expected="<<a.data_ptr<float>()[worst]<<" actual="<<b.data_ptr<float>()[worst]<<"\n";
                throw std::runtime_error("eager prefill differs from captured FFN");
            }
            auto f=result[1].to(tb::kFloat32);
            for(int64_t i=0;i<f.numel();++i)if(!std::isfinite(f.data_ptr<float>()[i]))
                throw std::runtime_error("nonfinite eager prefill output");
            ++cases;
            }
            if(tokens==512 && step==0) {
                std::ostringstream out;print_moe_expert_cache_stats(caches[1],out);std::smatch match;
                const auto stats=out.str();
                if(!std::regex_search(stats,match,std::regex("pipeline_direct_ram_bytes=(\\d+)")) || std::stoll(match[1])<=0)
                    throw std::runtime_error("prefill did not copy registered RAM directly");
                if(grouped && (!std::regex_search(stats,match,std::regex("pipeline_prefill_groups=(\\d+)")) || std::stoll(match[1])<3))
                    throw std::runtime_error("prefill did not reuse its streamed expert ring");
                const auto* phased=std::getenv("MFQ_MOE_PREFILL_LAYER_PHASED");
                if(layer_prefill && !layout_only && (!phased || std::atoi(phased)!=0) &&
                    (!std::regex_search(stats,match,std::regex("pipeline_phased_transfer_serves=(\\d+)")) || std::stoll(match[1])<=0))
                    throw std::runtime_error("whole-layer prefill did not partition Gate/Up and Down");
                ++direct_checks;
            }
        }
        forward={};caches={};
    }
    std::cout<<"prefill_wide_row_original_bits_checks="<<wide_row_bits_checks<<" PASS"<<std::endl;
    std::cout<<"prefill_grouped="<<grouped<<" prefill_graph_execution_cases="<<cases<<" direct_ram_checks="<<direct_checks<<" pool_retention_checks="<<pool_checks
             <<" canonical_oracle="<<oracle<<" same_layer_reuse_checks="<<reuse_checks<<" max_relative_rms="<<maximum_relative
             <<" cross_batch_checks="<<batch_checks<<" cross_batch_max_relative_rms="<<maximum_batch_relative
             <<" NINT/E8/D4, mixed G/U/Down, asynchronous layers sharing scratch, partial VRAM+RAM, changing/repeated routes, decode transition PASS\n";
}
