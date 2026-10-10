#pragma once
#include <chrono>
#include <iomanip>

inline void residency_copy_benchmark(const std::filesystem::path& root) {
    struct Environment {
        std::string old;
        Environment() {const auto* value=std::getenv("MFQ_MOE_RESIDENCY_BATCHED");if(value)old=value;}
        static void set(const char* value) {
#if defined(_WIN32)
            _putenv_s("MFQ_MOE_RESIDENCY_BATCHED",value);
#else
            if(*value)setenv("MFQ_MOE_RESIDENCY_BATCHED",value,1);else unsetenv("MFQ_MOE_RESIDENCY_BATCHED");
#endif
        }
        ~Environment(){set(old.c_str());}
    } environment;
    const std::vector<std::string> formats={"nint4","nint5","nint6","nint8","nvq1-s","nvq1-l",
        "nvq2j","nvq2j-l","nvq2j-xl","nvq3j","nvq3j-512","nvq3j-l"};
    struct Case {
        std::shared_ptr<MoeExpertCache> cache;
        std::vector<std::array<std::shared_ptr<MoeQuantRangeSource>,3>> sources;
        std::vector<MoeFfnForward> forwards;
        std::unique_ptr<MoeResidencyManager> manager;
        int step=0;
        double run() {
            const int expert=++step%3;
            const auto start=std::chrono::steady_clock::now();
            for(std::size_t layer=0;layer<sources.size();++layer)
                manager->after_layer(int(layer),std::vector<int32_t>(10,expert),1);
            manager->apply_pending();
            const auto elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
            // Tier identities are checked outside the timed migration. The
            // numerical/rollback oracle is the ordinary native regression.
            for(const auto& layer:sources)for(const auto& source:layer)for(int e=0;e<3;++e)
                if(source->gpu_resident(e)!=(e==expert) ||
                        source->host_cache()->contains(source->host_key(e))!=(e!=expert) ||
                        source->expert_disk_reads_after_preload())
                    throw std::runtime_error("timed residency migration changed tier ownership or read SSD");
            return elapsed;
        }
    };
    std::array<Case,2> cases;
    std::size_t projection_bytes=0;
    for(int mode=0;mode<2;++mode) {
        Environment::set(mode?"1":"0");auto& c=cases[mode];
        CudaExecutionContext execution;execution.config.moe_residency_warm=true;
        execution.config.moe_residency_projection_heat=true;
        execution.config.moe_direct_ram=true;
        execution.config.moe_host_physical_bytes=32ull<<30;
        c.sources.resize(formats.size());std::size_t gpu_bytes=0;
        for(std::size_t layer=0;layer<formats.size();++layer)for(int p=0;p<3;++p) {
            c.sources[layer][p]=source(root/(formats[layer]+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
            gpu_bytes+=bytes(c.sources[layer][p]->metadata()->pools[0]);
        }
        if(!mode)projection_bytes=gpu_bytes;
        execution.config.moe_host_cache_bytes=3*gpu_bytes;
        c.cache=make_moe_expert_cache(gpu_bytes,execution.config);
        for(std::size_t layer=0;layer<formats.size();++layer) {
            std::vector<std::shared_ptr<MfeWeight>> gates,downs;
            for(int p=0;p<3;++p) {
                auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(c.cache,
                    std::to_string(layer)+"/"+std::to_string(p),c.sources[layer][p],1,int(layer),std::to_string(p)));
                (p==2?downs:gates).push_back(std::move(weight));
            }
            c.forwards.push_back(make_moe_ffn_pipeline(gates,downs));
        }
        finalize_moe_expert_cache(c.cache);
        c.manager=std::make_unique<MoeResidencyManager>(c.cache.get());
    }
    for(auto& c:cases)for(int warm=0;warm<3;++warm)c.run();
    int rounds=1;
    while(true) {
        double elapsed=0;for(int round=0;round<rounds;++round)elapsed+=cases[0].run();
        if(elapsed>=50000)break;
        rounds*=2;
    }
    std::array<std::vector<double>,2> timings;
    std::array<MoeResidencyManager::Stats,2> before={cases[0].manager->stats(),cases[1].manager->stats()};
    for(int sample=0;sample<7;++sample)for(int order=0;order<2;++order) {
        const int mode=(sample+order)%2;double elapsed=0;
        for(int round=0;round<rounds;++round)elapsed+=cases[mode].run();
        timings[mode].push_back(elapsed/rounds);
    }
    const auto median=[](std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];};
    std::cout<<std::setprecision(10)<<"{\"scope\":\"complementary_expert_cache_migration\",\"formats\":12,\"projection_count\":36"
        <<",\"field_bytes_per_direction\":"<<projection_bytes<<",\"samples\":7,\"rounds\":"<<rounds
        <<",\"whole_model_loads\":0,\"source\":\"legal real-shape synthetic packed fixtures\",\"off_us\":"<<median(timings[0])
        <<",\"on_us\":"<<median(timings[1])<<",\"speedup\":"<<median(timings[0])/median(timings[1]);
    for(int mode=0;mode<2;++mode) {
        const auto stats=cases[mode].manager->stats();const auto& old=before[mode];const double count=rounds*7;
        if(stats.committed_projections-old.committed_projections!=uint64_t(count)*36)
            throw std::runtime_error("timed cache migration did not exchange every projection");
        std::cout<<",\""<<(mode?"on":"off")<<"_samples_us\":[";
        for(std::size_t i=0;i<timings[mode].size();++i)std::cout<<(i?",":"")<<timings[mode][i];
        std::cout<<"],\""<<(mode?"on":"off")<<"_prepare_us\":"<<(stats.prepare_ns-old.prepare_ns)/count/1000
            <<",\""<<(mode?"on":"off")<<"_join_us\":"<<(stats.join_ns-old.join_ns)/count/1000
            <<",\""<<(mode?"on":"off")<<"_dma_wait_us\":"<<(stats.dma_wait_ns-old.dma_wait_ns)/count/1000
            <<",\""<<(mode?"on":"off")<<"_backup_copies\":"<<(stats.backup_copies-old.backup_copies)/count
            <<",\""<<(mode?"on":"off")<<"_upload_copies\":"<<(stats.upload_copies-old.upload_copies)/count;
    }
    std::cout<<"}\n";
}
