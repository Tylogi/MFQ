#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
namespace mfq::cuda {
inline double kv_resident_fraction(int64_t context) {
    constexpr double x[]{0,131072,262144,1048576},y[]{1,.6,.3,.1};
    if(context<=0)return 1;if(context>=1048576)return .1;
    double h[3],d[3],m[4];
    for(int i=0;i<3;++i){h[i]=x[i+1]-x[i];d[i]=(y[i+1]-y[i])/h[i];}
    m[0]=(3*d[0]-d[1])/2;m[3]=0;
    for(int i=1;i<3;++i){const double w1=2*h[i]+h[i-1],w2=h[i]+2*h[i-1];m[i]=(w1+w2)/(w1/d[i-1]+w2/d[i]);}
    int i=0;while(context>x[i+1])++i;
    const double t=(context-x[i])/h[i],t2=t*t,t3=t2*t;
    return std::clamp((2*t3-3*t2+1)*y[i]+(t3-2*t2+t)*h[i]*m[i]+(-2*t3+3*t2)*y[i+1]+(t3-t2)*h[i]*m[i+1],y[i+1],y[i]);
}
}
