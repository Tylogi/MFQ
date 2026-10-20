#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace mfq::models {
// Same interpolation ramp and attention scaling used by the Metal Qwen path.
struct YarnScaling {
    double factor=1, beta_fast=32, beta_slow=1;
    std::int64_t original_context=0;
    bool enabled() const { return factor>1; }
    double attention_factor() const { return enabled() ? 1+0.1*std::log(factor) : 1; }
    double frequency(int pair,int dimension,double theta) const {
        const auto frequency=std::pow(theta,-2.0*pair/dimension);
        if(!enabled())return frequency;
        if(original_context<=0 || !std::isfinite(factor) || !std::isfinite(beta_fast) ||
            !std::isfinite(beta_slow) || beta_slow<=0 || beta_fast<=beta_slow || theta<=1)
            throw std::invalid_argument("invalid Qwen YaRN scaling");
        const auto correction=[&](double rotations) {
            return dimension*std::log(original_context/(rotations*2*std::acos(-1.0)))/(2*std::log(theta));
        };
        const auto low=std::max(0.0,std::floor(correction(beta_fast)));
        auto high=std::min(double(dimension-1),std::ceil(correction(beta_slow)));
        if(low==high)high+=0.001;
        const auto ramp=std::clamp((pair-low)/(high-low),0.0,1.0);
        return frequency*(1-ramp+ramp/factor);
    }
};
}
