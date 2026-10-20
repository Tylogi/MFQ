#pragma once
#include "cpu_cost_model.h"
#include "cpu_calibration.h"
#include <memory>
#include <string>


inline int cpu_calibration_checks() {
    mfq::MoeCpuCostModel model;mfq::MoeCpuCalibrationState state;
    const mfq::MoeCpuCostModel::Key key{1,2,3,1};
    std::promise<void> promise;
    auto ready=promise.get_future().share();
    mfq::start_moe_cpu_calibration(state,key,[ready]{ready.wait();},1000);
    if(!mfq::collect_moe_cpu_calibration(state,model,false) || model.observations(key))
        throw std::runtime_error("pending CPU calibration blocked or altered costs");
    bool rejected=false;
    try {mfq::start_moe_cpu_calibration(state,key,[]{});}catch(const std::logic_error&){rejected=true;}
    promise.set_value();
    if(!rejected)throw std::runtime_error("CPU calibration accepted concurrent work");
    mfq::collect_moe_cpu_calibration(state,model,true);
    if(state.busy() || model.observations(key)!=1 || state.jobs()!=1 || state.observations()!=1)
        throw std::runtime_error("CPU calibration lost its completed observation");
    for(int sample=0;sample<2;++sample) {
        mfq::start_moe_cpu_calibration(state,key,[]{},1000);
        mfq::collect_moe_cpu_calibration(state,model,true);
    }
    const auto measured=model.estimate(key);
    if(measured<1000 || state.observations()!=3 || state.work_ns()<3000)
        throw std::runtime_error("CPU background measurement lost preparation time");
    mfq::start_moe_cpu_calibration(state,key,[]{throw std::runtime_error("required calibration error");});
    rejected=false;
    try {mfq::collect_moe_cpu_calibration(state,model,true);}catch(const std::runtime_error& error) {
        rejected=std::string(error.what())=="required calibration error";
    }
    if(!rejected || state.busy())throw std::runtime_error("CPU worker error was swallowed");
    std::promise<double> invalid;state.pending=invalid.get_future();invalid.set_value(-1);
    rejected=false;
    try {mfq::collect_moe_cpu_calibration(state,model,true);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected || model.observations(key)!=3)
        throw std::runtime_error("invalid background timing corrupted CPU costs");
    rejected=false;
    try {mfq::start_moe_cpu_calibration(state,key,[]{},-1);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected || state.busy())throw std::runtime_error("invalid CPU measurement preparation accepted");
    std::weak_ptr<int> weak;
    {
        mfq::MoeCpuCalibrationState lifetime;
        auto owner=std::make_shared<int>(7);weak=owner;
        mfq::start_moe_cpu_calibration(lifetime,key,[owner]{if(*owner!=7)throw std::runtime_error("lost lease");});
        owner.reset();
    }
    if(!weak.expired())throw std::runtime_error("CPU calibration retained inputs after cleanup");
    mfq::reset_moe_cpu_calibration(state,model);
    if(state.jobs() || state.observations() || state.work_ns() || model.estimate(key)!=measured)
        throw std::runtime_error("CPU diagnostic reset changed observed costs");
    return 9;
}

inline int cpu_cost_checks() {
    mfq::MoeCpuCostModel model;
    const mfq::MoeCpuCostModel::Key key{17,17,23,1};
    int cases=0;
    if(model.estimate(key)!=0)throw std::runtime_error("unmeasured CPU work has a fabricated cost");++cases;
    model.observe(key,10000000,1);
    if(model.estimate(key)!=0)throw std::runtime_error("one cold observation established the rate");++cases;
    model.observe(key,1000000,1);
    if(model.estimate(key)!=0)throw std::runtime_error("CPU rate became ready before median calibration");++cases;
    model.observe(key,1000000,1);
    if(model.estimate(key)!=1000000)throw std::runtime_error("cold wake poisoned shared CPU median");++cases;
    for(int round=0;round<6;++round) {
        model.observe(key,1000000,1);
        if(model.estimate(key)!=1000000)throw std::runtime_error("consistent warm samples changed median");++cases;
    }
    for(int round=0;round<24;++round) {
        model.observe(key,round%5==0?20000000:2000000,2);
        if(model.estimate(key)!=1000000 || model.observations(key)>9)
            throw std::runtime_error("CPU pooled samples or rolling outliers changed the rate");
        ++cases;
    }
    for(int round=0;round<9;++round)model.observe(key,4000000,2);
    if(model.estimate(key)!=2000000)throw std::runtime_error("CPU cost failed to track sustained rate changes");++cases;
    for(int field=0;field<4;++field) {
        auto different=key;++different[field];
        if(model.estimate(different)!=0)throw std::runtime_error("different CPU format or row count reused a rate");
        ++cases;
    }
    for(double value:{0.0,-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected=false;
        try {model.observe(key,value,1);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected || model.estimate(key)!=2000000)throw std::runtime_error("invalid CPU timing corrupted history");
        ++cases;
    }
    for(std::size_t groups:{std::size_t(0),std::size_t(4)}) {
        auto invalid=key;if(groups)invalid[3]=0;
        bool rejected=false;
        try {model.observe(invalid,1.0,groups);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("empty CPU geometry accepted");++cases;
    }
    mfq::MoeCpuCostModel shared;
    int calibrations=0;
    // Forty-eight layer instances, three compatible FFN signatures. Actual
    // retained samples are shared; no per-layer measurement is fabricated.
    for(int layer=0;layer<48;++layer) {
        mfq::MoeCpuCostModel::Key same{std::uintptr_t(layer%3+1),7,9,1};
        if(!shared.estimate(same)){shared.observe(same,1000.0*(layer%3+1),1);++calibrations;}
    }
    if(calibrations!=9)throw std::runtime_error("compatible layers recalibrated independently");++cases;
    return cases+cpu_calibration_checks();
}
