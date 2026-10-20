#pragma once

static void benchmark_kv_speculative_lru() {
    namespace tb=mfq_tensor_backend;
    constexpr int layers=32,experts=32,entries=layers*experts,iterations=128,samples=7;
    Stream stream{0};KvExpertLoan loan;
    auto arena=tb::zeros({int64_t(entries)*KvExpertLoan::page},
        tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
    loan.add(arena,entries,KvExpertLoan::page,true);
    for(int layer=0;layer<layers;++layer)for(int expert=0;expert<experts;++expert) {
        bool hit=false;
        check(loan.pin_projection(100+layer,expert,{int64_t(KvExpertLoan::page)},stream.get(),hit,
            [](void*,const auto&) {})!=nullptr,"LRU benchmark allocation failed");
    }
    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    std::array<int32_t,40> ids{};
    for(int row=0;row<4;++row)for(int route=0;route<10;++route)ids[row*10+route]=row*4+route;
    const auto cycle=[&](int committed) {
        loan.begin_speculative(4);
        for(int layer=0;layer<layers;++layer) {
            loan.next_packet({100+layer,0,0},ids.data(),4,10,experts);
            for(int expert=0;expert<22;++expert) {
                std::array<std::size_t,8> offsets{};std::size_t bytes=0;
                check(loan.pin_existing_projection(100+layer,expert,{int64_t(KvExpertLoan::page)},offsets,bytes)!=nullptr,
                    "LRU benchmark lost a cached projection");
            }
        }
        const auto started=std::chrono::steady_clock::now();
        loan.commit_speculative(committed);
        return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
    };
    for(int committed:{1,3,4}) {
        for(int warm=0;warm<8;++warm)cycle(committed);
        std::vector<double> totals,commits;
        for(int sample=0;sample<samples;++sample) {
            double commit=0;const auto started=std::chrono::steady_clock::now();
            for(int step=0;step<iterations;++step)commit+=cycle(committed);
            totals.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count()/iterations);
            commits.push_back(commit/iterations);
        }
        std::sort(totals.begin(),totals.end());std::sort(commits.begin(),commits.end());
        std::cout<<"kv_lru_host_bench entries="<<entries<<" layers="<<layers<<" rows=4 committed="<<committed
            <<" samples="<<samples<<" iterations="<<iterations<<" median_window_us="<<totals[samples/2]
            <<" median_commit_us="<<commits[samples/2]<<" min_commit_us="<<commits.front()
            <<" max_commit_us="<<commits.back()<<'\n';
    }
    check(loan.speculative_stats().disabled==0,"LRU benchmark disabled tracking");
}

static void check_kv_speculative_lru() {
    namespace tb=mfq_tensor_backend;
    struct Fixture {
        Stream copy{0};KvExpertLoan loan;tb::Tensor arena;
        explicit Fixture(int pages=3) {
            arena=tb::zeros({int64_t(pages)*KvExpertLoan::page},
                tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
            loan.add(arena,pages,KvExpertLoan::page,true);
        }
        const void* put(int expert,int64_t source=17) {
            bool hit=false;
            const auto* pointer=loan.pin_projection(source,expert,{int64_t(KvExpertLoan::page)},copy.get(),hit,
                [&](void* destination,const auto&) {
                    MFQ_NATIVE_CUDA_CHECK(cudaMemsetAsync(destination,int((source+expert)&255),KvExpertLoan::page,copy.get()));
                });
            check(pointer!=nullptr,"speculative fixture exhausted protected pages");return pointer;
        }
        const void* get(int expert,int64_t source=17) const {
            return loan.find_for_exchange({source,expert,0},KvExpertLoan::page);
        }
        const void* hit(int expert,int64_t source=17) {
            std::array<std::size_t,8> offsets{};std::size_t bytes=0;
            const auto* pointer=loan.pin_existing_projection(source,expert,{int64_t(KvExpertLoan::page)},offsets,bytes);
            check(pointer && bytes==KvExpertLoan::page,"speculative fixture expected an existing projection");return pointer;
        }
        void packet(const std::vector<int32_t>& ids,int rows=3) {
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));
            check(ids.size()%rows==0,"invalid fixture packet");
            loan.next_packet({17,18,19},ids.data(),rows,int(ids.size()/rows),64);
        }
        void advance() {MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));loan.next_packet();}
        void bytes(int expert,int64_t source=17) {
            const auto* pointer=get(expert,source);check(pointer!=nullptr,"speculative payload was lost");
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy.get()));
            std::vector<uint8_t> values(KvExpertLoan::page);
            MFQ_NATIVE_CUDA_CHECK(cudaMemcpy(values.data(),pointer,values.size(),cudaMemcpyDeviceToHost));
            check(std::all_of(values.begin(),values.end(),[&](uint8_t value){return value==uint8_t(source+expert);}),
                "speculative LRU changed expert payload bytes");
        }
    };
    int cases=0;
    // Control reproduces an accepted hot expert being displaced by rejected
    // hits; transaction-aware recency must preserve that expert instead.
    for(bool tracking:{false,true})for(int committed:{1,2,3}) {
        Fixture f;for(int expert:{1,2,3})f.put(expert);
        const auto* accepted=f.get(1);
        if(tracking)f.loan.begin_speculative(3);
        f.packet({1,1,2,2,3,3});for(int expert:{1,2,3})f.hit(expert);
        f.loan.commit_speculative(committed);f.advance();f.put(4);
        const int evicted=!tracking || committed==3?1:committed==2?3:2;
        check(!f.get(evicted),"speculative cache chose the wrong eviction victim");
        if(tracking && committed<3)check(f.get(1)==accepted,"rejected hits displaced the accepted expert");
        for(int expert:{1,2,3,4})if(expert!=evicted)f.bytes(expert);
        if(tracking)check(f.loan.speculative_stats().kept==uint64_t(committed),"committed cache touch count changed");
        ++cases;
    }
    {
        Fixture f;for(int expert:{1,2,3})f.put(expert);
        f.loan.begin_speculative(3);f.packet({1,2,3});f.hit(1);f.hit(2);f.loan.discard_speculative();
        f.advance();f.put(4);
        check(!f.get(1) && f.get(3),"cancelled cache hits changed pre-window eviction order");
        check(f.loan.speculative_stats().cancels==1,"cancelled cache transaction was not counted");++cases;
    }
    {
        Fixture f(4);for(int expert:{1,2,3})f.put(expert);
        f.loan.begin_speculative(3);f.packet({1,4,2});f.hit(1);const auto* rejected=f.put(4);f.hit(2);
        f.loan.commit_speculative(1);
        check(f.get(4)==rejected,"rejected insertion was freed before its packet fence");f.bytes(4);
        f.advance();f.put(5);
        check(!f.get(4) && f.get(1) && f.get(2) && f.get(3),"new rejected expert stayed hotter than existing experts");
        check(f.loan.speculative_stats().cold_insertions==1,"new rejected cache insertion was not recorded");++cases;
    }
    {
        Fixture f;for(int expert:{1,2,3})f.put(expert);
        f.loan.begin_speculative(3);f.packet({1,2,1});f.hit(2);f.hit(1);f.loan.commit_speculative(1);
        f.advance();f.put(4);check(f.get(1) && !f.get(2),"expert used by accepted and rejected rows became cold");++cases;
    }
    {
        Fixture f;f.put(1);f.put(2);f.put(3,23);
        f.loan.begin_speculative(3);f.packet({1,2,3});f.hit(1);f.hit(2);f.hit(3,23);f.loan.commit_speculative(1);
        f.advance();f.put(4);
        check(f.get(3,23) && f.get(1) && !f.get(2),"another source inherited the target's speculative priority");
        f.bytes(3,23);++cases;
    }
    {
        Fixture f(2);f.put(1);f.put(2);f.loan.begin_speculative(3);f.packet({1,2,3});f.hit(2);
        f.packet({1,2,3});f.put(3);f.packet({1,2,3});f.put(4);f.packet({1,2,3});f.put(2);
        f.loan.commit_speculative(1);
        check(f.loan.speculative_stats().cold_insertions==1 && !f.loan.speculative_stats().reverted_hits,
            "evicted/reinserted transaction key reused an earlier cache lifetime");f.bytes(2);++cases;
    }
    {
        Fixture f;for(int expert:{1,2,3})f.put(expert);
        f.loan.begin_speculative(3);f.packet({1,2,3});f.hit(2);f.loan.retire_expert(17,2,1);
        f.loan.commit_speculative(1);f.advance();check(!f.get(2),"retired speculative expert survived its packet");++cases;
    }
    {
        Fixture f;for(int expert:{1,2,3})f.put(expert);
        f.loan.begin_speculative(3);f.packet({1,2,3});f.hit(2);f.loan.ensure(3);f.loan.commit_speculative(1);
        check(!f.loan.bytes(),"KV reclamation retained a speculative expert");++cases;
    }
    for(int committed:{31,32}) {
        Fixture f;for(int expert:{1,2,3})f.put(expert);
        std::vector<int32_t> ids(32,1);ids.back()=2;
        f.loan.begin_speculative(32);f.packet(ids,32);f.hit(1);f.hit(2);f.loan.commit_speculative(committed);
        f.advance();f.put(4);check(!f.get(committed==32?3:2),"32-row accepted-prefix mask changed eviction");++cases;
    }
    {
        Fixture f;for(int expert:{1,2,3})f.put(expert);
        f.loan.begin_speculative(3);f.loan.speculative_packet({17,18,19},nullptr,2,1,64);
        f.packet({1,2,3});for(int expert:{1,2,3})f.hit(expert);f.loan.commit_speculative(1);
        f.advance();f.put(4);
        check(!f.get(1) && f.loan.speculative_stats().disabled==1,"unsupported row geometry changed normal LRU behavior");++cases;
    }
    {
        Fixture f;bool rejected=false;
        try{f.loan.begin_speculative(33);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected,"oversized speculative row mask was accepted");
        f.loan.begin_speculative(3);rejected=false;
        try{f.loan.begin_speculative(2);}catch(const std::logic_error&){rejected=true;}
        check(rejected,"nested speculative LRU transaction was accepted");f.loan.discard_speculative();++cases;
    }
    std::cout<<"KV speculative LRU priority/payload/prefix/cancel/source/reinsert/retire/growth/mask exact PASS cases="<<cases<<'\n';
}
