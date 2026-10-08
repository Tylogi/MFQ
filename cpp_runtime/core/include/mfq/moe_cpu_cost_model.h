#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace mfq {
class MoeCpuCostModel {
public:
    // Three projection compatibility identities plus the repeated-row count.
    using Key=std::array<std::uintptr_t,4>;
private:
    struct History {
        std::array<double,9> samples{};
        std::size_t count=0,next=0;
    };
    std::map<Key,History> history_;
    std::uint64_t samples_=0;
    mutable std::uint64_t queries_=0,ready_queries_=0;
public:
    double estimate(const Key& key)const {
        ++queries_;
        const auto found=history_.find(key);
        if(found==history_.end() || found->second.count<3)return 0;
        ++ready_queries_;
        const auto& h=found->second;auto sorted=h.samples;
        std::sort(sorted.begin(),sorted.begin()+h.count);
        return h.count%2 ? sorted[h.count/2] : (sorted[h.count/2-1]+sorted[h.count/2])/2;
    }
    void observe(const Key& key,double elapsed_ns,std::size_t groups) {
        if(!groups || !std::isfinite(elapsed_ns) || elapsed_ns<=0 || !key[3])
            throw std::invalid_argument("CPU dispatch observation needs measured positive work");
        auto& h=history_[key];h.samples[h.next]=elapsed_ns/groups;
        h.next=(h.next+1)%h.samples.size();h.count=std::min(h.count+1,h.samples.size());
        ++samples_;
    }
    std::size_t observations(const Key& key)const {
        const auto found=history_.find(key);return found==history_.end()?0:found->second.count;
    }
    std::size_t keys()const noexcept {return history_.size();}
    std::uint64_t samples()const noexcept {return samples_;}
    std::uint64_t queries()const noexcept {return queries_;}
    std::uint64_t ready_queries()const noexcept {return ready_queries_;}
};
} // namespace mfq
