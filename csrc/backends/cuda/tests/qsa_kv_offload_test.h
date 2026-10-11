#pragma once

void qsa_kv_offload_case() {
    const auto metric = [](const std::shared_ptr<KvOffloadStore>& store, const char* key) {
        for (const auto& value : store->metrics()) if (value.first == key) return value.second;
        throw std::runtime_error(std::string("missing KV metric ") + key);
    };
    KvOffloadConfig config;
    config.gpu_budget_bytes = 16 << 10; config.buffer_bytes = 8 << 10;
    config.ram_budget_bytes = 16 << 10;
    auto store = std::make_shared<KvOffloadStore>(config);
    auto source = data({1024,32},19,.4f);
    KvOffloadSequence sequence(store,32,64);
    sequence.append(source.narrow(0,0,192));
    store->flush();
    if (metric(store,"qsa_kv_ram_bytes") != 4096 || metric(store,"qsa_kv_ssd_bytes") != 0)
        throw std::runtime_error("GPU eviction must enter RAM before SSD");
    sequence.append(source.narrow(0,192,832));
    store->flush();
    if (metric(store,"qsa_kv_resident_bytes") > 8192 || metric(store,"qsa_kv_ram_bytes") > 16384 ||
        metric(store,"qsa_kv_ssd_bytes") <= 0)
        throw std::runtime_error("KV tier limits or SSD overflow are incorrect");
    equal(sequence.gather({900,1,70,900,513}), source.index_select(0,tb::tensor(std::vector<int64_t>{900,1,70,900,513}).to(tb::kCUDA)),"KV three-tier gather",0,0);
    sequence.truncate(67);
    sequence.append(source.narrow(0,67,13));
    equal(sequence.range(0,80),source.narrow(0,0,80),"KV truncate and partial-tail replay",0,0);
    sequence.reset();store->flush();
    if (sequence.position() || metric(store,"qsa_kv_resident_bytes") || metric(store,"qsa_kv_ram_bytes") || metric(store,"qsa_kv_ssd_bytes"))
        throw std::runtime_error("KV reset must release all three tiers");
    config.ram_budget_bytes = 0;
    store = std::make_shared<KvOffloadStore>(config);
    KvOffloadSequence direct(store,32,64);
    direct.append(source.narrow(0,0,192));store->flush();
    if (metric(store,"qsa_kv_ram_bytes") || !metric(store,"qsa_kv_ssd_bytes"))
        throw std::runtime_error("zero RAM budget must spill directly to SSD");
    equal(direct.range(0,64),source.narrow(0,0,64),"KV direct SSD restore",0,0);

    for (int width : {8,256}) {
        CudaExecutionContext execution;
        constexpr int hidden=32,kv=2,iw=8,ih=2,pool=4,maximum=384,budget=64;
        const int heads=4;
        auto rotary=std::make_shared<RotaryEmbedding>(8,maximum,10000.);
        auto norm=data({width},1,0),inorm=data({iw},1,0);
        QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
            linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,inorm,inorm};
        QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
        config.buffer_bytes=1 << 20;config.gpu_budget_bytes=config.buffer_bytes+(8 << 10);config.ram_budget_bytes=8 << 10;
        store=std::make_shared<KvOffloadStore>(config);
        Qsa actual(weights,cfg,rotary,true,true,store),oracle(weights,cfg,rotary);
        auto inputs=data({1,maximum,hidden},17,.4f);
        int64_t offset=0;
        for(int tokens:{16,49,97,128,1}) {
            auto positions=tb::arange(offset,offset+tokens,inputs.options().dtype(tb::kInt64));
            auto history=tb::arange(offset+tokens,positions.options());
            auto x=inputs.narrow(1,offset,tokens);
            auto expected=oracle.forward(execution,x,positions,history,true);
            auto result=actual.forward(execution,x,positions,history,true);
            equal(result,expected,"QSA streamed dense/sparse prefill and decode");offset+=tokens;
        }
        actual.truncate(67);oracle.truncate(67);
        auto positions=tb::arange(67,80,inputs.options().dtype(tb::kInt64));
        auto history=tb::arange(80,positions.options());auto x=inputs.narrow(1,67,13);
        equal(actual.forward(execution,x,positions,history,true),oracle.forward(execution,x,positions,history,true),"QSA streamed truncate/replay");
        store->flush();
        if (!metric(store,"qsa_kv_ssd_read_bytes") || !metric(store,"qsa_kv_h2d_bytes"))
            throw std::runtime_error("QSA parity must exercise RAM and SSD restoration");
        actual.reset();store->flush();
        if (metric(store,"qsa_kv_resident_bytes") || metric(store,"qsa_kv_ram_bytes") || metric(store,"qsa_kv_ssd_bytes"))
            throw std::runtime_error("QSA reset leaked stored KV");
    }
    std::cout << "CUDA KV GPU -> RAM LRU -> SSD overflow, exact block restore and QSA oracle parity PASS\n";
}
