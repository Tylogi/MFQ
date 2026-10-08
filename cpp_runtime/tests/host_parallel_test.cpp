#include "mfq/host_parallel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void coverage(int threads, int count) {
    std::vector<std::atomic<int>> visits(count);
    for (auto& visit : visits) visit.store(0);
    mfq::host_parallel_for(17, 17 + count, 3, threads,
        [&](std::int64_t begin, std::int64_t end) {
            for (auto i = begin; i < end; ++i) ++visits[i-17];
        });
    for (const auto& visit : visits) require(visit == 1, "host work omitted or duplicated");
}

void baseline(std::int64_t end, int threads,
    const std::function<void(std::int64_t, std::int64_t)>& work) {
    std::atomic<std::int64_t> next{0};
    std::vector<std::thread> workers;
    for (int i=0; i<threads; ++i) workers.emplace_back([&] {
        for (;;) {
            const auto begin = next.fetch_add(1, std::memory_order_relaxed);
            if (begin >= end) return;
            work(begin, begin+1);
        }
    });
    for (auto& worker : workers) worker.join();
}

void benchmark() {
    const int threads = std::max(1u, std::thread::hardware_concurrency());
    constexpr int outputs = 640, width = 2560, operations = 144;
    std::vector<float> weights(outputs * width), input(width), result(outputs), reference;
    for (int i=0; i<outputs*width; ++i) weights[i] = float(i%29-14)*0.03125f;
    for (int i=0; i<width; ++i) input[i] = float(i%17-8)*0.0625f;
    auto work = [&](std::int64_t begin, std::int64_t end) {
        for (auto row=begin; row<end; ++row) {
            float total=0;
            for (int column=0; column<width; ++column)
                total += weights[row*width+column]*input[column];
            result[row]=total;
        }
    };
    baseline(outputs, threads, work);
    reference=result;
    mfq::host_parallel_for(0, outputs, 1, threads, work);
    require(result==reference, "dispatch changed expert projection results");
    for (int round=0; round<3; ++round) {
        auto start=std::chrono::steady_clock::now();
        for (int i=0; i<operations; ++i) baseline(outputs, threads, work);
        auto middle=std::chrono::steady_clock::now();
        for (int i=0; i<operations; ++i) mfq::host_parallel_for(0, outputs, 1, threads, work);
        auto end=std::chrono::steady_clock::now();
        const double old_ms=std::chrono::duration<double,std::milli>(middle-start).count();
        const double new_ms=std::chrono::duration<double,std::milli>(end-middle).count();
        require(result==reference, "benchmark result differs");
        std::cout << "{\"round\":" << round << ",\"threads\":" << threads
            << ",\"shape\":[640,2560],\"projections\":" << operations
            << ",\"baseline_ms\":" << old_ms << ",\"persistent_ms\":" << new_ms
            << ",\"speedup\":" << old_ms/new_ms << "}\n";
    }
}

void profile_checks() {
    mfq::HostParallelPool pool;
    const auto check=[](const mfq::HostParallelProfile& p,int rows,int participants) {
        require(!p.failed && p.rows==static_cast<std::uint64_t>(rows),"profile lost completed rows");
        require(p.participants==participants && p.callbacks>0,"profile dispatch counts differ");
        require(p.work_ns>=p.caller_work_ns && p.wall_ns>=p.caller_work_ns,"profile work timing invalid");
        require(p.worker_arrivals==static_cast<std::uint64_t>(participants-1),"profile lost worker arrivals");
        require(p.worker_wake_max_ns<=p.wall_ns,"profile worker timestamp outside dispatch");
    };
    std::atomic<int> completed{0};
    mfq::HostParallelProfile serial;
    pool.run(11,18,1,1,[&](auto b,auto e){completed+=static_cast<int>(e-b);},&serial);
    check(serial,7,1);
    for(int i=0;i<100;++i) {
        mfq::HostParallelProfile outer;
        pool.run(0,37,1,6,[&](auto b,auto e) {
            for(auto row=b;row<e;++row) {
                mfq::HostParallelProfile inner;
                pool.run(0,7,1,6,[&](auto a,auto z){completed+=static_cast<int>(z-a);},&inner);
                check(inner,7,1);
            }
        },&outer);
        check(outer,37,6);
        pool.run(0,19,1,6,[](auto,auto){}); // No stale profile pointer on unobserved dispatches.
    }
    require(completed==7+100*37*7,"profile changed nested results");
    auto client=[&] {
        for(int i=0;i<100;++i) {
            mfq::HostParallelProfile p;
            pool.run(0,83,1,6,[&](auto b,auto e){completed+=static_cast<int>(e-b);},&p);
            check(p,83,6);
        }
    };
    std::thread first(client),second(client);first.join();second.join();
    require(completed==7+100*37*7+200*83,"profile changed concurrent results");
    mfq::HostParallelProfile failed;
    bool rejected=false;
    try {pool.run(0,640,1,6,[](auto,auto){throw std::runtime_error("observed failure");},&failed);}
    catch(const std::runtime_error&) {rejected=true;}
    require(rejected && failed.failed && failed.wall_ns>0,"profile swallowed callback failure");
    mfq::HostParallelProfile recovered;
    pool.run(0,640,1,6,[](auto,auto){},&recovered);check(recovered,640,6);
    pool.run(0,0,1,6,[](auto,auto){throw std::runtime_error("empty profile work");},&recovered);
    require(recovered.rows==0 && recovered.wall_ns==0 && recovered.participants==0,"empty profile retained previous counts");
    std::cout<<"Host pool profile: exact counts, nested/concurrent isolation, failure propagation and recovery PASS\n";
}
void callback_lifetime_checks() {
    mfq::HostParallelPool pool({},false);
    std::atomic<int> visited{0};
    int expected_visited=0;
    for(int i=0;i<2000;++i) {
        std::atomic<bool> alive{true};
        std::atomic<int> callbacks{0};
        std::vector<std::atomic<int>> outer_visits(37);
        for(auto& visit:outer_visits)visit=0;
        mfq::HostParallelProfile profile;
        pool.run(0,37,1,6,[&](auto begin,auto end) {
            if(!alive.load())throw std::runtime_error("retired worker used an expired callback");
            ++callbacks;
            for(auto row=begin;row<end;++row)++outer_visits[row];
            visited+=int(end-begin);
            pool.run(0,3,1,6,[&](auto b,auto e){visited+=int(e-b);});
        },&profile);
        alive=false;
        require(profile.rows==37 && profile.worker_arrivals<=5,"retired dispatch lost rows or counted idle arrivals");
        require(profile.callbacks==static_cast<std::uint64_t>(callbacks.load()),"retired dispatch callback accounting differs");
        for(const auto& visit:outer_visits)require(visit==1,"retired dispatch omitted or duplicated an outer row");
        expected_visited+=37+3*callbacks.load();
        require(visited==expected_visited,"retired dispatch changed nested row coverage");
    }
    // Every callback owns the object until it finishes, including exceptions.
    bool rejected=false;
    try{pool.run(0,640,1,6,[](auto,auto){throw std::runtime_error("retired dispatch failure");});}
    catch(const std::runtime_error&){rejected=true;}
    require(rejected,"retired dispatch lost a callback failure");
    std::vector<std::atomic<int>> visits(640);
    pool.run(0,640,1,6,[&](auto begin,auto end){for(auto i=begin;i<end;++i)++visits[i];});
    for(const auto& visit:visits)require(visit==1,"retired dispatch recovery duplicated rows");
    mfq::HostParallelPool broken([](auto){throw std::runtime_error("retired initialization failure");},false);
    rejected=false;
    try{broken.run(0,640,1,6,[](auto,auto){});}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"retired pool ignored worker initialization failure");
    std::cout<<"Host pool retired workers: callback lifetime, coverage, nested work and failures PASS\n";
}
} // namespace

int main(int argc, char**) {
    try {
        profile_checks();
        callback_lifetime_checks();
        for (int repeat=0; repeat<100; ++repeat)
            for (int threads=1; threads<=12; ++threads) coverage(threads, 37+repeat);
        std::vector<std::thread> callers;
        for (int caller=0; caller<4; ++caller) callers.emplace_back([&] {
            for (int repeat=0; repeat<100; ++repeat) coverage(12, 640);
        });
        for (auto& caller : callers) caller.join();
        std::atomic<int> nested{0};
        mfq::host_parallel_for(0, 12, 1, 12, [&](auto begin, auto end) {
            for (auto i=begin; i<end; ++i)
                mfq::host_parallel_for(0, 7, 1, 12, [&](auto b, auto e) { nested += int(e-b); });
        });
        require(nested==84, "nested work deadlocked or was omitted");
        bool rejected=false;
        try {
            mfq::host_parallel_for(0, 640, 1, 12, [](auto, auto) { throw std::runtime_error("injected"); });
        } catch (const std::runtime_error&) { rejected=true; }
        require(rejected, "worker error did not propagate");
        coverage(12, 640);
        mfq::HostParallelPool broken([](std::size_t) { throw std::runtime_error("worker init failed"); });
        rejected = false;
        try { broken.run(0, 640, 1, 4, [](auto, auto) {}); }
        catch (const std::runtime_error& error) { rejected = std::string(error.what()) == "worker init failed"; }
        require(rejected, "worker initialization failure did not reach the caller");
        mfq::host_parallel_for(0, 0, 1, 12, [](auto, auto) { throw std::runtime_error("empty work invoked"); });
        std::cout << "Host pool: coverage, concurrency, nested work and failure recovery passed\n";
        if (argc>1) benchmark();
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
