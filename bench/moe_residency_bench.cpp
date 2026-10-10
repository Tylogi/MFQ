#include "packed_bench_utils.h"
#include "residency_rank_reference.h"
#include <array>
#include <chrono>
#include <fstream>
#include <sstream>

namespace {
using namespace mfq::bench;
constexpr int layers=48,experts=512;
struct Routes {int layer,tokens;std::vector<int> ids;};
struct Snapshot {
    std::vector<float> heat;
    std::vector<std::array<int,3>> resident;
    float value(const ResidencyExpert& e)const {return heat[std::size_t(e.bundle)*experts+e.expert];}
    int state(const ResidencyExpert& e)const {
        const auto& s=resident[std::size_t(e.bundle)*experts+e.expert];
        return e.projection>=0 ? s[e.projection] : s[0]==s[1] && s[1]==s[2] ? s[0] : -1;
    }
};
std::vector<Routes> read_routes(const char* path) {
    std::ifstream file(path);if(!file)throw std::runtime_error("cannot read retained routing trace");
    std::vector<Routes> routes;std::string line;
    while(std::getline(file,line)) {
        if(line.empty())continue;
        Routes row{};std::size_t entries,cpu;std::istringstream text(line);
        if(!(text>>row.layer>>row.tokens>>entries>>cpu) || row.layer<0 || row.layer>=layers || row.tokens<1)
            throw std::runtime_error("invalid retained route header");
        row.ids.resize(entries);
        for(auto& id:row.ids)if(!(text>>id) || id<0 || id>=experts)throw std::runtime_error("invalid expert route");
        for(std::size_t i=0;i<cpu;++i){int id;if(!(text>>id))throw std::runtime_error("missing CPU trace field");}
        std::string extra;if(text>>extra)throw std::runtime_error("unexpected route trace fields");
        routes.push_back(std::move(row));
    }
    if(routes.empty() || routes.back().layer!=layers-1)throw std::runtime_error("incomplete route trace");
    return routes;
}
volatile std::uint64_t result_checksum=0;
void consume(const std::vector<mfq::MoeResidencyRanker::Pair>& pairs) {
    std::uint64_t value=pairs.size();
    if(!pairs.empty())value+=pairs.front().incoming.expert+std::uint64_t(pairs.back().outgoing.bundle)*experts;
    result_checksum=result_checksum+value;
}
template<class Fn> double measure(Fn fn,std::size_t rounds,std::size_t windows) {
    const auto begin=std::chrono::steady_clock::now();
    for(std::size_t i=0;i<rounds;++i)fn();
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/(rounds*windows);
}
void benchmark(const std::vector<Routes>& routes,bool projections) {
    std::vector<ResidencyEntry> entries;
    for(int b=0;b<layers;++b)for(int e=0;e<experts;++e)for(int p=0;p<(projections?3:1);++p)
        entries.push_back({{b,e,projections?p:-1},std::size_t((p==2?8:0)+(e+b)%8)});
    mfq::MoeResidencyRanker ranker(residency_layout(entries,16));
    Snapshot current;current.heat.resize(layers*experts);current.resident.resize(layers*experts);
    for(int b=0;b<layers;++b)for(int e=0;e<experts;++e)for(int p=0;p<3;++p)
        current.resident[std::size_t(b)*experts+e][p]=((e*7+b*11+(projections?p*13:0))%100)<30;
    std::vector<Snapshot> snapshots;
    std::size_t exact=0,candidates=0;
    for(const auto& route:routes) {
        for(int e:route.ids)current.heat[std::size_t(route.layer)*experts+e]+=1.0f;
        if(route.layer!=layers-1)continue;
        const auto heat=[&](const ResidencyExpert& e){return current.value(e);};
        const auto state=[&](const ResidencyExpert& e){return current.state(e);};
        const auto reference=residency_reference(entries,heat,state);
        if(!same_residency_pairs(reference,ranker.rank(heat,state)))throw std::runtime_error("ordered migration candidates differ");
        ++exact;candidates+=reference.size();snapshots.push_back(current);
        // Synthetic initial tier ownership, then deterministic exchanges from
        // the unchanged selector. No weight tensors or whole model are loaded.
        for(const auto& pair:reference) {
            auto& in=current.resident[std::size_t(pair.incoming.bundle)*experts+pair.incoming.expert];
            auto& out=current.resident[std::size_t(pair.outgoing.bundle)*experts+pair.outgoing.expert];
            if(pair.incoming.projection<0){in.fill(1);out.fill(0);}
            else {in[pair.incoming.projection]=1;out[pair.outgoing.projection]=0;}
        }
        for(auto& value:current.heat)value*=0.7f;
    }
    auto off=[&] {
        for(const auto& s:snapshots)consume(residency_reference(entries,
            [&](const ResidencyExpert& e){return s.value(e);},[&](const ResidencyExpert& e){return s.state(e);}));
    };
    auto on=[&] {
        for(const auto& s:snapshots)consume(ranker.rank(
            [&](const ResidencyExpert& e){return s.value(e);},[&](const ResidencyExpert& e){return s.state(e);}));
    };
    off();on();
    std::size_t rounds=1;
    while(measure(off,rounds,snapshots.size())*rounds*snapshots.size()<50000)rounds*=2;
    std::vector<double> before,after;
    for(int sample=0;sample<7;++sample) {
        if(sample%2){after.push_back(measure(on,rounds,snapshots.size()));before.push_back(measure(off,rounds,snapshots.size()));}
        else {before.push_back(measure(off,rounds,snapshots.size()));after.push_back(measure(on,rounds,snapshots.size()));}
    }
    for(const auto& s:snapshots) {
        const auto heat=[&](const ResidencyExpert& e){return s.value(e);};
        const auto state=[&](const ResidencyExpert& e){return s.state(e);};
        if(!same_residency_pairs(residency_reference(entries,heat,state),ranker.rank(heat,state)))
            throw std::runtime_error("post-timing candidate verification failed");
    }
    std::cout<<std::setprecision(10)<<"{\"scope\":\"host_residency_ranking\",\"mode\":\""
        <<(projections?"projection":"bundle")<<"\",\"layers\":"<<layers<<",\"experts_per_layer\":"<<experts
        <<",\"topology_entries\":"<<entries.size()<<",\"compatibility_groups\":16,\"trace_rows\":"<<routes.size()
        <<",\"windows\":"<<snapshots.size()<<",\"ordered_exact_cases\":"<<exact<<",\"candidate_total\":"<<candidates
        <<",\"whole_model_loads\":0,\"synthetic_initial_residency\":true,\"samples\":7,\"rounds\":"<<rounds;
    timing(before,after);std::cout<<"}\n";
}
}
int main(int argc,char** argv)try {
    if(argc!=2)throw std::invalid_argument("usage: mfq-moe-residency-bench retained-dispatch-record.txt");
    const auto routes=read_routes(argv[1]);benchmark(routes,false);benchmark(routes,true);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
