#pragma once
#include "../../bench/residency_rank_reference.h"
#include <array>
#include <cstdint>

inline int residency_rank_checks() {
    using namespace mfq::bench;
    std::uint32_t random=719;
    const auto next=[&] {random^=random<<13;random^=random>>17;random^=random<<5;return random;};
    int cases=0;
    for(bool projections:{false,true})for(int layers:{1,3,48})for(int experts:{1,7,512}) {
        std::vector<ResidencyEntry> entries;
        for(int b=0;b<layers;++b)for(int e=0;e<experts;++e)
            for(int p=0;p<(projections?3:1);++p)
                entries.push_back({{b,e,projections?p:-1},std::size_t((b+e+p)%11)});
        mfq::MoeResidencyRanker ranker(residency_layout(entries,12));
        std::vector<float> heat(std::size_t(layers)*experts);
        std::vector<std::array<int,3>> states(heat.size());
        auto get_heat=[&](const ResidencyExpert& e){return heat[std::size_t(e.bundle)*experts+e.expert];};
        auto get_state=[&](const ResidencyExpert& e) {
            const auto& s=states[std::size_t(e.bundle)*experts+e.expert];
            return e.projection>=0 ? s[e.projection] : s[0]==s[1] && s[1]==s[2] ? s[0] : -1;
        };
        // Reuse one selector through empty/cold/hot/split, ties, threshold
        // boundaries and changing tier ownership.
        for(int round=0;round<24;++round) {
            for(std::size_t i=0;i<heat.size();++i) {
                heat[i]=round%6==0?0.0f:round%6==1?2.0f:float(next()%25)*0.5f;
                for(auto& state:states[i])state=round%6==2?0:round%6==3?1:int(next()%2);
            }
            const auto expected=residency_reference(entries,get_heat,get_state);
            if(!same_residency_pairs(expected,ranker.rank(get_heat,get_state)))
                throw std::runtime_error("cached expert ranking changed ordered candidates");
            ++cases;
        }
    }
    mfq::MoeResidencyRanker empty({});
    if(!empty.rank([](auto){return 0.0f;},[](auto){return 0;}).empty())
        throw std::runtime_error("empty expert topology produced candidates");
    bool rejected=false;
    try {mfq::MoeResidencyRanker invalid({{{0,0,3}}});}
    catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("invalid expert topology accepted");
    return cases+2;
}
