#include "mfq/kernels/cuda/mfe_ffn.h"
#include "mfq/kernels/cuda/mfe_ffn_launch.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

static void setting(const char* value) {
#ifdef _WIN32
    _putenv_s("MFQ_MFE_E8_NARROW",value);
#else
    if(*value)setenv("MFQ_MFE_E8_NARROW",value,1);
    else unsetenv("MFQ_MFE_E8_NARROW");
#endif
}

int main()try {
    using namespace mfq::cuda;
    int device=0,major=0,minor=0;
    if(cudaGetDevice(&device)!=cudaSuccess ||
       cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device)!=cudaSuccess ||
       cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device)!=cudaSuccess)
        throw std::runtime_error("device query failed");
    const bool sm86=major==8 && minor==6;
    int checks=0;
    const auto verify=[&](const MfeFfnBatch& b,bool down,bool expected) {
        if(mfe_ffn_e8_narrow_requested(b,down)!=expected)
            throw std::runtime_error("MFE narrow default selection mismatch");
        ++checks;
    };
    MfeFfnBatch baseline;
    baseline.tokens=1;baseline.routes=10;
    baseline.input_width=baseline.output_width=2560;baseline.intermediate=640;
    MfePackedProjection shared[3];uint32_t ready=1;
    for(bool asynchronous:{false,true})for(bool include_shared:{false,true}) {
        baseline.plan_ready=asynchronous?&ready:nullptr;
        baseline.shared=include_shared?shared:nullptr;
        baseline.shared_intermediate=include_shared?640:0;
        setting("");
        for(bool down:{false,true})verify(baseline,down,sm86);
        for(int member=0;member<6;++member) {
            auto changed=baseline;
            switch(member) {
                case 0:changed.tokens=3;break;
                case 1:changed.routes=11;break;
                case 2:changed.input_width=2561;break;
                case 3:changed.output_width=2559;break;
                case 4:changed.intermediate=641;break;
                case 5:changed.shared=shared;changed.shared_intermediate=641;break;
            }
            for(bool down:{false,true})verify(changed,down,false);
        }
        for(const char* value:{"0","1","2","3","invalid"}) {
            setting(value);
            for(bool down:{false,true})verify(baseline,down,value[0]=='1' || value[0]==(down?'3':'2'));
        }
    }
    setting("");
    std::cout<<"mfe_e8_default_dispatch_checks="<<checks<<" sm86="<<sm86<<" PASS\n";
    return 0;
}catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
