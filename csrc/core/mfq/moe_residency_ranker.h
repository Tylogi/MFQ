#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace mfq {
// Compatibility groups are fixed by the loaded formats and field layouts.
// Only heat and residency change between completed inference windows.
class MoeResidencyRanker {
public:
    struct Expert { int bundle=0, expert=0, projection=-1; };
    struct Pair { Expert incoming,outgoing; float gain=0; };
private:
    struct Item { Expert expert; float heat=0; };
    struct Group {
        std::vector<Expert> entries;
        std::vector<Item> cold,hot;
    };
    std::vector<Group> groups_;
    std::vector<Pair> pairs_;
    static bool colder(const Item& a,const Item& b) {
        return a.heat!=b.heat ? a.heat<b.heat :
            std::tie(a.expert.bundle,a.expert.expert,a.expert.projection)<
            std::tie(b.expert.bundle,b.expert.expert,b.expert.projection);
    }
public:
    explicit MoeResidencyRanker(std::vector<std::vector<Expert>> layout) {
        groups_.reserve(layout.size());
        for(auto& entries:layout) {
            for(const auto& e:entries)
                if(e.bundle<0 || e.expert<0 || e.projection<-1 || e.projection>2)
                    throw std::invalid_argument("invalid expert ranking topology");
            Group group; group.entries=std::move(entries);
            groups_.push_back(std::move(group));
        }
    }
    // state: 1 fully resident, 0 fully in RAM, -1 split across tiers.
    // Partial bundles participate through projection entries only.
    template<class Heat,class State>
    std::vector<Pair>& rank(Heat heat,State state) {
        pairs_.clear();
        for(auto& group:groups_) {
            group.cold.clear();group.hot.clear();
            for(const auto& e:group.entries) {
                const Item item{e,heat(e)};const int placement=state(e);
                if(placement==1)group.hot.push_back(item);
                else if(placement==0 && item.heat>=2.0f)group.cold.push_back(item);
            }
            std::sort(group.cold.begin(),group.cold.end(),
                [](const Item& a,const Item& b){return colder(b,a);});
            const auto count=std::min(group.cold.size(),group.hot.size());
            std::partial_sort(group.hot.begin(),group.hot.begin()+count,group.hot.end(),colder);
            for(std::size_t i=0;i<count;++i) {
                const float gain=group.cold[i].heat-group.hot[i].heat;
                if(gain<1.5f)break;
                pairs_.push_back({group.cold[i].expert,group.hot[i].expert,gain});
            }
        }
        std::sort(pairs_.begin(),pairs_.end(),
            [](const Pair& a,const Pair& b){return a.gain>b.gain;});
        return pairs_;
    }
};
} // namespace mfq
