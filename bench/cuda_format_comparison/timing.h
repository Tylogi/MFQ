#pragma once
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace format_bench {
inline void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Case {
    std::string label, model, tensor;
    int expert=0, n=0, k=0;
    double file_bpw=0;
    std::string name() const { return label+"_"+std::to_string(n)+"x"+std::to_string(k); }
};
inline std::vector<Case> read_cases(const char* path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open cases manifest");
    std::vector<Case> result;
    for (std::string line; std::getline(file,line);) {
        if (line.empty()) continue;
        std::istringstream row(line); Case c; std::string expert,n,k,bpw;
        if (!std::getline(row,c.label,'\t') || !std::getline(row,c.model,'\t') ||
            !std::getline(row,c.tensor,'\t') || !std::getline(row,expert,'\t') ||
            !std::getline(row,n,'\t') || !std::getline(row,k,'\t') || !std::getline(row,bpw))
            throw std::runtime_error("invalid cases manifest row");
        c.expert=std::stoi(expert);c.n=std::stoi(n);c.k=std::stoi(k);c.file_bpw=std::stod(bpw);
        if (c.n<=0 || c.k<=0 || c.expert<0 || !std::isfinite(c.file_bpw))
            throw std::runtime_error("invalid case dimensions");
        result.push_back(std::move(c));
    }
    if(result.empty()) throw std::runtime_error("empty cases manifest");
    return result;
}
inline std::vector<float> input(int m,int k) {
    std::vector<float> values(std::size_t(m)*k);
    for(std::size_t i=0;i<values.size();++i)
        values[i]=std::sin(float(i+7)*.193f)*.31f + std::cos(float(i+11)*.071f)*.11f;
    return values;
}
template<class Replay> std::vector<double> measure(Replay replay,cudaStream_t stream,std::size_t calls) {
    cudaEvent_t begin=nullptr,end=nullptr;
    check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    auto time=[&](std::size_t repeats) {
        check(cudaEventRecord(begin,stream));
        for(std::size_t i=0;i<repeats;++i)replay();
        check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
        float elapsed=0;check(cudaEventElapsedTime(&elapsed,begin,end));return double(elapsed);
    };
    // GPU clocks may idle during IQ quantization or range loading between cases.
    // Sustain work before calibrating; keep all seven samples in the report.
    double warm_ms=0;int warm_rounds=0;
    while(warm_ms<200. || warm_rounds<8){warm_ms+=time(8);++warm_rounds;}
    std::size_t repeats=1;
    // One scheduling stall must not choose an undersized sampling window.
    while(std::min({time(repeats),time(repeats),time(repeats)})<20.)repeats*=2;
    std::vector<double> values;
    for(int sample=0;sample<7;++sample)values.push_back(time(repeats)*1000./double(repeats*calls));
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));return values;
}
inline double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());return values[values.size()/2];
}
inline void profile_kernels(cudaGraph_t captured,cudaStream_t stream,std::size_t calls,const std::string& label) {
    std::size_t count=0;check(cudaGraphGetNodes(captured,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(captured,nodes.data(),&count));
    std::map<void*,std::vector<cudaKernelNodeParams>> kernels;
    for(auto node:nodes) {
        cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel)continue;
        cudaKernelNodeParams p{};check(cudaGraphKernelNodeGetParams(node,&p));kernels[p.func].push_back(p);
    }
    for(const auto& [function,parameters]:kernels) {
        cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;cudaGraphNode_t previous=nullptr;
        check(cudaGraphCreate(&graph,0));
        for(const auto& p:parameters) {
            cudaGraphNode_t node=nullptr;
            check(cudaGraphAddKernelNode(&node,graph,previous?&previous:nullptr,previous?1:0,&p));previous=node;
        }
        check(cudaGraphInstantiate(&exec,graph,0));check(cudaGraphUpload(exec,stream));
        const auto samples=measure([&](){check(cudaGraphLaunch(exec,stream));},stream,calls);
        cudaFuncAttributes attr{};check(cudaFuncGetAttributes(&attr,function));
        const auto& p=parameters.front();
        std::cout<<"KERNEL "<<label<<" function="<<function<<" nodes_per_call="<<double(parameters.size())/calls
            <<" grid="<<p.gridDim.x<<','<<p.gridDim.y<<','<<p.gridDim.z
            <<" block="<<p.blockDim.x<<','<<p.blockDim.y<<','<<p.blockDim.z
            <<" regs="<<attr.numRegs<<" us="<<median(samples)<<'\n'<<std::flush;
        check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
    }
}
inline void report(const Case& c,const char* backend,const std::string& format,int m,int stored_k,
                   std::size_t bytes,std::size_t banks,int l2,const std::vector<double>& samples,int routes=1) {
    const double us=median(samples);
    std::cout<<std::setprecision(10)<<"{\"case\":\""<<c.name()<<"\",\"backend\":\""<<backend
        <<"\",\"format\":\""<<format<<"\",\"M\":"<<m<<",\"N\":"<<c.n<<",\"K\":"<<c.k
        <<",\"stored_K\":"<<stored_k<<",\"file_bpw\":"<<c.file_bpw
        <<",\"routes\":"<<routes<<",\"weight_bytes\":"<<bytes<<",\"gpu_bpw\":"<<double(bytes)*8./(double(c.n)*c.k*routes)
        <<",\"banks\":"<<banks<<",\"working_bytes\":"<<bytes*banks<<",\"l2_bytes\":"<<l2
        <<",\"scope\":\"activation_quantization_and_matmul_cuda_graph\",\"us\":"<<us
        <<",\"effective_GBs\":"<<double(bytes)/(us*1000.)<<",\"samples_us\":[";
    for(std::size_t i=0;i<samples.size();++i)std::cout<<(i?",":"")<<samples[i];
    std::cout<<"]}\n"<<std::flush;
}
} // namespace format_bench
