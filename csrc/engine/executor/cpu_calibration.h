#pragma once
#include "cpu_cost_model.h"
#include "mfq/moe/cpu_measurement.h"
#include <chrono>
#include <functional>
#include <utility>

namespace mfq {
inline void start_moe_cpu_calibration(MoeCpuCalibrationState& state,
        MoeCpuCostModel::Key key,std::function<void()> work,double preparation_ns=0) {
    if(state.busy())throw std::logic_error("CPU calibration already has pending work");
    state.pending=cpu::measure_work(std::move(work),preparation_ns);
    state.key=key;
    ++state.jobs_total;
}
// An unfinished measurement occupies the CPU; it cannot precede inference work.
inline bool collect_moe_cpu_calibration(MoeCpuCalibrationState& state,MoeCpuCostModel& model,bool wait) {
    if(!state.busy())return false;
    if(!wait && state.pending.wait_for(std::chrono::seconds(0))!=std::future_status::ready)return true;
    const auto elapsed=state.pending.get();
    model.observe(state.key,elapsed,1);
    ++state.observations_total;
    state.elapsed_ns+=elapsed;
    return false;
}
inline void reset_moe_cpu_calibration(MoeCpuCalibrationState& state,MoeCpuCostModel& model) {
    collect_moe_cpu_calibration(state,model,true);
    state.jobs_total=state.observations_total=0;
    state.elapsed_ns=0;
}
} // namespace mfq
