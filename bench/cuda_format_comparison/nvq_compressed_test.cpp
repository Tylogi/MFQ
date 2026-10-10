#include "vq.h"
#include <array>
#include <cstring>
#include <iostream>

namespace tb=mfq_tensor_backend;
static void clear_flag(const char* name) {
#ifdef _WIN32
    _putenv_s(name,"");
#else
    unsetenv(name);
#endif
}
static void check_payload(const tb::Tensor& tensor,const std::vector<uint8_t>& expected) {
    const auto host=tensor.to(tb::kCPU).contiguous();
    if(host.numel()!=int64_t(expected.size()) ||
       (!expected.empty() && std::memcmp(host.data_ptr(),expected.data(),expected.size())))
        throw std::runtime_error("default NVQ upload changed compressed payload");
}
int main()try {
    for(const char* name:{"MFQ_NVQ2_EXEC","MFQ_DISABLE_NVQ2_EXEC","MFQ_DISABLE_NIQ2_EXEC",
            "MFQ_NVQ_EXTENDED_GROUP_EXEC","MFQ_NVQ1_GROUP_RECORDS","MFQ_NVQ1_INTEGER_DELTA"})clear_flag(name);
    const std::array<CudaExecutionConfig,2> configs={CudaExecutionConfig{},load_cuda_execution_config()};
    for(const auto& config:configs)if(config.nvq2_exec || config.nvq_extended_group_exec)
        throw std::runtime_error("default NVQ execution layout expands weights");
    int devices=0;const auto status=cudaGetDeviceCount(&devices);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && !devices))return 77;
    MFQ_CUDA_CHECK(status);
    auto context=mfq::cuda::default_context(0);auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    int checks=0;
    for(const auto& config:configs)for(int format:{1,2,5,8,10,12,13,14,15})for(int width:{1,25,640,2560}) {
        NvqCpu c;c.format=format;c.gs=24;c.sub_bits=format==1?3:4;c.out=17;
        c.neuron_len=width;c.ng=(width+23)/24;c.nsign=(width+7)/8;
        const int dims=(format==10 || format==12 || format==15)?4:8;
        c.nvec=(width+dims-1)/dims;c.shape={c.out,width};
        const int bits=format==1?11:(format==8 || format==12)?9:(format==13 || format==15)?10:format==14?12:8;
        c.indices_packed.resize((int64_t(c.out)*c.nvec*bits+7)/8);
        c.aux_packed.resize((int64_t(c.out)*((format==1 || format==8)?c.ng:c.nsign*7)+7)/8);
        c.sub_scale_packed.resize((int64_t(c.out)*c.ng*c.sub_bits+7)/8);
        for(auto* payload:{&c.indices_packed,&c.aux_packed,&c.sub_scale_packed})
            for(size_t i=0;i<payload->size();++i)(*payload)[i]=uint8_t(i*37+format*11);
        c.neuron_scale_h.assign(c.out,0x3c00u);
        const int entries=1<<bits;
        c.codebook.resize(format==1 || format==2 || format==8 ? entries*dims : 64+4*entries*dims);
        if(format==10) {c.codebook[1]=2;c.codebook[3]=1;}
        auto weight=to_device_nvq(c,true,config);
        if(weight.kernel_format!=(format==10?11:format) || weight.sign_mode!=0 ||
                weight.decode_records.defined() || weight.integer_codebook.defined())
            throw std::runtime_error("default NVQ upload selected an expanded execution layout");
        check_payload(weight.indices_packed,c.indices_packed);
        check_payload(weight.aux_packed,c.aux_packed);
        check_payload(weight.sub_scale_packed,c.sub_scale_packed);
        ++checks;
    }
    std::cout<<"Default compressed NVQ upload: "<<checks<<" checks, payload bytes/bits unchanged PASS\n";
    return 0;
}catch(const std::exception& error) {std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
