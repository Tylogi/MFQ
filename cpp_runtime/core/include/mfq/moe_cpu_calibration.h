#pragma once
#include "mfq/moe_cpu_cost_model.h"
#include <chrono>
#include <functional>
#include <future>
#include <utility>

namespace mfq {
// The scheduler owns both this state and the cost model. A worker only returns
// an observation; it never mutates routing, costs, or expert cache ownership.
class MoeCpuCalibration {
public:
    struct Observation {
        MoeCpuCostModel::Key key{};
        double elapsed_ns=0;
    };
private:
    std::future<Observation> pending_;
    std::uint64_t jobs_=0,observations_=0;
    double work_ns_=0;
public:
    bool busy()const noexcept {return pending_.valid();}
    void start(std::function<Observation()> work) {
        if(busy())throw std::logic_error("CPU calibration already has pending work");
        pending_=std::async(std::launch::async,std::move(work));
        ++jobs_;
    }
    // Return true only while an unfinished observation still occupies the CPU.
    bool collect(MoeCpuCostModel& model,bool wait) {
        if(!busy())return false;
        if(!wait && pending_.wait_for(std::chrono::seconds(0))!=std::future_status::ready)return true;
        const auto result=pending_.get();
        model.observe(result.key,result.elapsed_ns,1);
        ++observations_;work_ns_+=result.elapsed_ns;
        return false;
    }
    // Teardown waits without replacing an already reported execution error.
    // Captured immutable inputs and RAM leases survive until the worker ends.
    void wait() {if(busy())pending_.wait();}
    void reset(MoeCpuCostModel& model) {
        collect(model,true);
        jobs_=observations_=0;work_ns_=0;
    }
    std::uint64_t jobs()const noexcept {return jobs_;}
    std::uint64_t observations()const noexcept {return observations_;}
    double work_ns()const noexcept {return work_ns_;}
};
} // namespace mfq
