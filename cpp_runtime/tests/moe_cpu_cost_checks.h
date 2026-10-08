#pragma once
#include "mfq/moe_cpu_cost_model.h"
#include "mfq/moe_cpu_calibration.h"
#include <memory>
#include <string>


inline int cpu_calibration_checks() {
    using Observation=mfq::MoeCpuCalibration::Observation;
    mfq::MoeCpuCostModel model;mfq::MoeCpuCalibration state;
    const mfq::MoeCpuCostModel::Key key{1,2,3,1};
    std::promise<Observation> promise;
    auto result=std::make_shared<std::future<Observation>>(promise.get_future());
    state.start([result]{return result->get();});
    if(!state.collect(model,false) || model.observations(key))
        throw std::runtime_error("pending CPU calibration blocked or altered costs");
    bool rejected=false;
    try {state.start([]{return Observation{};});}catch(const std::logic_error&){rejected=true;}
    promise.set_value({key,1000});
    if(!rejected)throw std::runtime_error("CPU calibration accepted concurrent work");
    state.collect(model,true);
    if(state.busy() || model.observations(key)!=1 || state.jobs()!=1 || state.observations()!=1)
        throw std::runtime_error("CPU calibration lost its completed observation");
    for(double value:{1200.,1100.}) {
        state.start([key,value]{return Observation{key,value};});state.collect(model,true);
    }
    if(model.estimate(key)!=1100 || state.observations()!=3)
        throw std::runtime_error("CPU background median differs from measured work");
    state.start([]()->Observation {throw std::runtime_error("required calibration error");});
    rejected=false;
    try {state.collect(model,true);}catch(const std::runtime_error& error) {
        rejected=std::string(error.what())=="required calibration error";
    }
    if(!rejected || state.busy())throw std::runtime_error("CPU worker error was swallowed");
    state.start([key]{return Observation{key,-1};});rejected=false;
    try {state.collect(model,true);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected || model.observations(key)!=3)
        throw std::runtime_error("invalid background timing corrupted CPU costs");
    std::weak_ptr<int> weak;
    {
        mfq::MoeCpuCalibration lifetime;
        auto owner=std::make_shared<int>(7);weak=owner;
        lifetime.start([owner,key]{return Observation{key,double(*owner)};});
        owner.reset();
    }
    if(!weak.expired())throw std::runtime_error("CPU calibration retained inputs after cleanup");
    state.reset(model);
    if(state.jobs() || state.observations() || state.work_ns() || model.estimate(key)!=1100)
        throw std::runtime_error("CPU diagnostic reset changed observed costs");
    return 8;
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
