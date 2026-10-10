void verify_nvq_prefill_k_tiles(const std::filesystem::path& root,bool cohorts=false,bool defaults=false) {
    struct Environment {
        const char* name;
        std::string previous;
        bool present=false;
        explicit Environment(const char* key):name(key){if(const auto* p=std::getenv(name)){previous=p;present=true;}}
        void set(const char* p) const {
#ifdef _WIN32
            _putenv_s(name,p);
#else
            if(*p)setenv(name,p,1);else unsetenv(name);
#endif
        }
        ~Environment(){set(present?previous.c_str():"");}
    } environment(cohorts?"MFQ_NVQ_PREFILL_COHORTS":"MFQ_NVQ_PREFILL_NARROW_G2");
    CudaExecutionContext execution;
    constexpr int experts=512,routes=10;
    int cases=0;
    const auto formats=cohorts?std::vector<std::string>{"mixed"}:
        std::vector<std::string>{"nvq2j","nvq3j","nvq3j-512","nvq2j-xl","nvq3j-l"};
    const auto shapes=cohorts?std::vector<std::pair<int,int>>{{640,2560},{2560,640}}:
        std::vector<std::pair<int,int>>{{640,2560}};
    for(const auto& format:formats)for(const auto shape:shapes) {
        const int out=shape.first,width=shape.second;
        auto runtime=std::make_shared<MixedMoeRuntime>();
        runtime->n_experts=experts;runtime->out_per_expert=out;runtime->neuron_len=width;
        const auto members=cohorts?std::vector<std::string>{"nvq2j","nvq3j","nvq3j-512","nvq2j-xl","nvq3j-l","nvq1-s","nvq1-l"}:
            std::vector<std::string>{format};
        int member_index=0;
        for(const auto& member:members) {
        auto model=mfq::open_model_source((root/(member+"-"+std::to_string(out)+"-"+std::to_string(width)+".mfq")).string());
        auto single=make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),true,execution.config);
        if(single->pools.size()!=1 || single->pools[0].family!=MixedMoeFamily::Nvq)
            throw std::runtime_error("NVQ K-tile check requires a single NVQ fixture pool per member");
        auto pool=std::move(single->pools[0]);
        if(!pool.nvq.dense_groups) {
            const auto device=pool.nvq.indices_packed.device();
            pool.nvq.indices_packed=pool.nvq.indices_packed.cpu();
            pool.nvq.aux_packed=pool.nvq.aux_packed.cpu();
            pool.nvq.sub_scale_packed=pool.nvq.sub_scale_packed.cpu();
            (void)prepare_nvq_dense_groups(pool.nvq);
            pool.nvq.indices_packed=pool.nvq.indices_packed.to(device);
            pool.nvq.aux_packed=pool.nvq.aux_packed.to(device);
            pool.nvq.sub_scale_packed=pool.nvq.sub_scale_packed.to(device);
        }
        // Give every logical expert separate compressed storage so a three-
        // expert fixture does not turn the throughput test into an L2-only run.
        const int pool_experts=(experts+int(members.size())-1-member_index)/int(members.size());
        const int repetitions=(pool_experts+pool.local_experts-1)/pool.local_experts;
        const auto replicate=[&](tb::Tensor& tensor) {
            if(tensor.numel())tensor=tb::cat(std::vector<tb::Tensor>(repetitions,tensor),0);
        };
        replicate(pool.nvq.indices_packed);replicate(pool.nvq.aux_packed);
        replicate(pool.nvq.sub_scale_packed);replicate(pool.nvq.neuron_scale);
        pool.nvq.out*=repetitions;pool.local_experts*=repetitions;
        std::vector<int32_t> slots(experts,-1);
        for(int e=member_index;e<experts;e+=int(members.size()))slots[e]=e/int(members.size());
        pool.expert_local=tb::tensor(slots).to(tb::kCUDA);
        runtime->pools.push_back(std::move(pool));++member_index;
        }
        initialize_mixed_nvq_dispatch(*runtime,execution.config,true);
        if(!runtime->nvq_dispatch)throw std::runtime_error("NVQ K-tile fixture dispatch was not prepared");
        auto& dispatch=*runtime->nvq_dispatch;
        for(int tokens:{4096,8192})for(int pattern:{0,1}) {
            const int pairs=tokens*routes;
            std::vector<float> values(tokens*width);
            for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.017f)*.13f;
            auto input=tb::tensor(values).reshape({tokens,width}).to(tb::kFloat16).to(tb::kCUDA);
            std::vector<std::vector<int32_t>> selected(experts);
            for(int p=0;p<pairs;++p) {
                const int e=(p*37)%experts;
                selected[e].push_back(p);
            }
            if(pattern==1) {
                // Include empty experts and route counts just across 128/256.
                for(int e=384;e<experts;++e) {
                    selected[e-384].insert(selected[e-384].end(),selected[e].begin(),selected[e].end());
                    selected[e].clear();
                }
            }
            std::vector<int32_t> compact,bounds(experts+1),tile_bounds(experts+1),tile_experts(pairs,-1);
            int tiles=0;
            for(int e=0;e<experts;++e) {
                bounds[e]=int(compact.size());tile_bounds[e]=tiles;
                compact.insert(compact.end(),selected[e].begin(),selected[e].end());
                for(size_t first=0;first<selected[e].size();first+=128)tile_experts[tiles++]=e;
            }
            bounds.back()=int(compact.size());tile_bounds.back()=tiles;
            auto ids_dst=tb::tensor(compact).to(tb::kCUDA),expert_bounds=tb::tensor(bounds).to(tb::kCUDA);
            auto tiles_bounds=tb::tensor(tile_bounds).to(tb::kCUDA),tiles_experts=tb::tensor(tile_experts).to(tb::kCUDA);
            auto output=tb::empty({tokens,routes,out},input.options());
            const auto run=[&] {
                nvq_moe_grouped_matmul_hetero_f16_cuda(dispatch.weight_ptrs,dispatch.weight_sizes,dispatch.pool_params,
                    dispatch.expert_pool,dispatch.expert_local,input,experts,out,width,128,output,
                    ids_dst,expert_bounds,tiles_bounds,tiles_experts,int(dispatch.f16_format_group),false);
            };
            std::vector<float> times[2];tb::Tensor reference;
            for(int repeat=0;repeat<8;++repeat)for(int order=0;order<2;++order) {
                const int candidate=order^(repeat&1);
                environment.set(candidate?(defaults?"":"1"):"0");
                run();run();mfq_cuda_synchronize();
                cudaEvent_t start,stop;MFQ_CUDA_CHECK(cudaEventCreate(&start));MFQ_CUDA_CHECK(cudaEventCreate(&stop));
                MFQ_CUDA_CHECK(cudaEventRecord(start,mfq_current_cuda_stream()));
                for(int i=0;i<3;++i)run();
                MFQ_CUDA_CHECK(cudaEventRecord(stop,mfq_current_cuda_stream()));MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
                float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));cudaEventDestroy(start);cudaEventDestroy(stop);
                auto actual=output.cpu().contiguous();
                if(!reference.defined())reference=actual.clone();
                if(std::memcmp(actual.data_ptr(),reference.data_ptr(),actual.nbytes()))
                    throw std::runtime_error("NVQ G2 changed original output bits: "+format);
                if(repeat>=2)times[candidate].push_back(ms/3);
                ++cases;
            }
            for(auto& samples:times)std::sort(samples.begin(),samples.end());
            std::cout<<"nvq_prefill_k_pair format="<<format<<" out="<<out<<" K="<<width<<" tokens="<<tokens<<" routes="<<routes
                <<" experts="<<experts<<" pattern="<<pattern<<" compressed_fixture_replication=1"
                <<" cohorts="<<cohorts<<" format_group="<<int(dispatch.f16_format_group)
                <<" old_ms="<<times[0][times[0].size()/2]<<" candidate_ms="<<times[1][times[1].size()/2]
                <<" output_bits_equal=1"<<std::endl;
        }
    }
    std::cout<<"nvq_prefill_k_tile_exact_cases="<<cases<<" PASS"<<std::endl;
}
