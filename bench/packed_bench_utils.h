#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::bench {
struct Settings {
    int samples=7, threads=6;
    double target_ms=20;
    std::size_t working_bytes=64u*1024u*1024u;
    bool all_formats=false;
};
inline Settings settings(int argc,char** argv) {
    Settings s;
    for(int i=1;i<argc;++i) {
        const std::string option=argv[i];
        if(option=="--all-formats") {s.all_formats=true;continue;}
        if(i+1==argc)throw std::invalid_argument("missing value for "+option);
        const std::string value=argv[++i];
        if(option=="--samples")s.samples=std::stoi(value);
        else if(option=="--threads")s.threads=std::stoi(value);
        else if(option=="--target-ms")s.target_ms=std::stod(value);
        else if(option=="--working-mib")s.working_bytes=std::stoull(value)*1024u*1024u;
        else throw std::invalid_argument("unknown option "+option);
    }
    if(s.samples<3 || s.threads<1 || !(s.target_ms>0) || !s.working_bytes)
        throw std::invalid_argument("invalid benchmark geometry");
    return s;
}
inline double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());
    const auto n=values.size();
    if(!n)throw std::invalid_argument("empty benchmark samples");
    return n%2 ? values[n/2] : (values[n/2-1]+values[n/2])/2;
}
inline void values(const std::vector<double>& samples) {
    std::cout<<'[';
    for(std::size_t i=0;i<samples.size();++i)std::cout<<(i ? "," : "")<<samples[i];
    std::cout<<']';
}
inline std::uint32_t random_word(std::uint32_t& state) {
    state^=state<<13;state^=state>>17;state^=state<<5;return state;
}
inline void fill(std::vector<std::uint8_t>& bytes,std::uint32_t& state) {
    for(auto& value:bytes)value=static_cast<std::uint8_t>(random_word(state));
}
inline void timing(const std::vector<double>& off,const std::vector<double>& on) {
    std::cout<<",\"off_us\":"<<median(off)<<",\"on_us\":"<<median(on)
             <<",\"speedup\":"<<median(off)/median(on)
             <<",\"off_samples_us\":";values(off);
    std::cout<<",\"on_samples_us\":";values(on);
}
} // namespace mfq::bench
