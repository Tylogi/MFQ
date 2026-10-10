#pragma once
#include "mfq/moe_dispatch_plan.h"
#include <stdexcept>

inline int dispatch_reuse_checks() {
    using K=mfq::MoeDispatchKind;
    mfq::MoeDispatchPlan reused;
    mfq::MoeDispatchWorkspace workspace;
    const auto equal=[](const auto& a,const auto& b) {
        if(a.kinds!=b.kinds || a.groups.size()!=b.groups.size())return false;
        for(std::size_t i=0;i<a.groups.size();++i)
            if(a.groups[i].expert!=b.groups[i].expert || a.groups[i].kind!=b.groups[i].kind ||
                    a.groups[i].positions!=b.groups[i].positions)return false;
        return true;
    };
    unsigned state=0x12345678u;
    const auto random=[&]() {state=1664525u*state+1013904223u;return state;};
    int cases=0;
    for(int experts:{1,5,17,512})for(int count:{0,1,10,30,80,320})for(int round=0;round<15;++round) {
        std::vector<std::int32_t> ids(count);
        for(auto& id:ids)id=round%5==0?int(random()%std::min(experts,3)):int(random()%experts);
        const auto hot=[&](int e) {return (unsigned(e)+unsigned(round))%4==0;};
        const auto eligible=[&](int e) {return (unsigned(e)+unsigned(round))%3!=0;};
        for(std::size_t quota:{std::size_t(0),std::size_t(1),std::size_t(3),std::size_t(1000)}) {
            std::vector<int> old_calls,new_calls;
            const auto expected=mfq::plan_moe_dispatch(ids,experts,quota,
                [&](int e){old_calls.push_back(e);return hot(e);},
                [&](int e){old_calls.push_back(-e-1);return eligible(e);});
            mfq::plan_moe_dispatch_reuse(reused,workspace,ids,experts,quota,
                [&](int e){new_calls.push_back(e);return hot(e);},
                [&](int e){new_calls.push_back(-e-1);return eligible(e);});
            if(!equal(expected,reused) || old_calls!=new_calls)
                throw std::runtime_error("reused plan differs in ordered groups, positions, quota or callback order");
            ++cases;
        }
        // The pipeline first classifies resident groups, then knows its quota.
        const auto initial=mfq::plan_moe_dispatch(ids,experts,0,hot,[](int){return true;});
        std::size_t misses=0;for(const auto& g:initial.groups)misses+=g.kind==K::Cpu;
        for(double fraction:{0.0,0.2,0.55,1.0}) {
            const auto quota=std::size_t(misses*fraction);
            const auto expected=mfq::plan_moe_dispatch(ids,experts,quota,hot,[](int){return true;});
            mfq::plan_moe_dispatch_reuse(reused,workspace,ids,experts,0,hot,[](int){return true;});
            mfq::assign_moe_transfer_quota(reused,quota,[](int){return true;});
            if(!equal(expected,reused))throw std::runtime_error("reused cold quota changes fractional dispatch");
            ++cases;
        }
    }
    const auto before=reused;const auto before_map=workspace.group_for;
    for(const auto ids:{std::vector<std::int32_t>{0,-1},std::vector<std::int32_t>{0,512}}) {
        int callbacks=0;bool rejected=false;
        try {mfq::plan_moe_dispatch_reuse(reused,workspace,ids,512,10,
            [&](int){++callbacks;return true;},[&](int){++callbacks;return true;});}
        catch(const std::out_of_range&) {rejected=true;}
        if(!rejected || callbacks || !equal(before,reused) || workspace.group_for!=before_map)
            throw std::runtime_error("invalid reused route changed plan or called placement");
        ++cases;
    }
    std::vector<std::int32_t> ids(10);for(int i=0;i<10;++i)ids[i]=i;
    const auto cold=[](int){return false;};const auto all=[](int){return true;};
    mfq::plan_moe_dispatch_reuse(reused,workspace,ids,512,10,cold,all);
    const auto* groups=reused.groups.data();const auto* kinds=reused.kinds.data();
    const auto* mapping=workspace.group_for.data();std::vector<const std::size_t*> positions;
    for(const auto& group:reused.groups)positions.push_back(group.positions.data());
    for(int round=0;round<100;++round) {
        for(int i=0;i<10;++i)ids[i]=(i+round*10)%512;
        mfq::plan_moe_dispatch_reuse(reused,workspace,ids,512,10,cold,all);
        if(groups!=reused.groups.data() || kinds!=reused.kinds.data() || mapping!=workspace.group_for.data())
            throw std::runtime_error("fixed top-k reuse reallocated plan storage");
        for(int i=0;i<10;++i)if(positions[i]!=reused.groups[i].positions.data())
            throw std::runtime_error("fixed top-k reuse reallocated position storage");
        ++cases;
    }
    return cases;
}
