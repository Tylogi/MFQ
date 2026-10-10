#pragma once
#include "mfq/moe_residency_ranker.h"
#include <map>

namespace mfq::bench {
using ResidencyExpert=MoeResidencyRanker::Expert;
struct ResidencyEntry {ResidencyExpert expert;std::size_t group;};

inline std::vector<std::vector<ResidencyExpert>> residency_layout(
        const std::vector<ResidencyEntry>& entries,std::size_t groups) {
    std::vector<std::vector<ResidencyExpert>> result(groups);
    for(const auto& item:entries)result.at(item.group).push_back(item.expert);
    return result;
}
// Retained MFQ selector: rebuild compatibility buckets for every window.
template<class Heat,class State>
std::vector<MoeResidencyRanker::Pair> residency_reference(
        const std::vector<ResidencyEntry>& entries,Heat heat,State state) {
    struct Item {ResidencyExpert expert;float heat;};
    struct Group {std::vector<Item> cold,hot;};
    std::map<std::size_t,Group> groups;
    for(const auto& entry:entries) {
        auto& group=groups[entry.group];const auto& e=entry.expert;
        const Item item{e,heat(e)};const int placement=state(e);
        if(placement==1)group.hot.push_back(item);
        else if(placement==0 && item.heat>=2.0f)group.cold.push_back(item);
    }
    auto colder=[](const Item& a,const Item& b) {
        return a.heat!=b.heat ? a.heat<b.heat :
            std::tie(a.expert.bundle,a.expert.expert,a.expert.projection)<
            std::tie(b.expert.bundle,b.expert.expert,b.expert.projection);
    };
    std::vector<MoeResidencyRanker::Pair> pairs;
    for(auto& entry:groups) {
        auto& group=entry.second;
        std::sort(group.cold.begin(),group.cold.end(),[&](const Item& a,const Item& b){return colder(b,a);});
        const auto count=std::min(group.cold.size(),group.hot.size());
        std::partial_sort(group.hot.begin(),group.hot.begin()+count,group.hot.end(),colder);
        for(std::size_t i=0;i<count;++i) {
            const float gain=group.cold[i].heat-group.hot[i].heat;
            if(gain<1.5f)break;
            pairs.push_back({group.cold[i].expert,group.hot[i].expert,gain});
        }
    }
    std::sort(pairs.begin(),pairs.end(),[](const auto& a,const auto& b){return a.gain>b.gain;});
    return pairs;
}
inline bool same_residency_pairs(const std::vector<MoeResidencyRanker::Pair>& a,
        const std::vector<MoeResidencyRanker::Pair>& b) {
    if(a.size()!=b.size())return false;
    const auto identity=[](const ResidencyExpert& e){return std::tie(e.bundle,e.expert,e.projection);};
    for(std::size_t i=0;i<a.size();++i)
        if(identity(a[i].incoming)!=identity(b[i].incoming) ||
           identity(a[i].outgoing)!=identity(b[i].outgoing) || a[i].gain!=b[i].gain)return false;
    return true;
}
} // namespace mfq::bench
