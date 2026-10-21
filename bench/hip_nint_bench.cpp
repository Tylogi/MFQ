#include "packed_bench_utils.h"
#include "mfq_hip_quant_ops.h"
#include "hip_execution.h"
#include "quant_linear.h"

#include <hip/hip_runtime.h>
#include <cstring>
#include <memory>

namespace {
using namespace mfq::cuda;
void check(hipError_t status) {
    if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));
}
NintWeight weight(int rows,int width,int bits,int gs,uint32_t seed) {
    NintCpu c;
    c.out=rows;c.gs=gs;c.ng=(width+gs-1)/gs;c.neuron_len=width;
    c.shape={rows,width};c.axis=0;c.bits=bits;c.sub_bits=6;c.format_version=2;
    c.row_q_bits.assign(rows,bits);c.row_sub_bits.assign(rows,0);
    c.row_q_bit_offsets.resize(rows);
    for(int row=0;row<rows;++row)c.row_q_bit_offsets[row]=int64_t(row)*c.ng*gs*bits;
    c.q_packed.resize((uint64_t(rows)*c.ng*gs*bits+7)/8+8);
    c.sub_scale.resize(size_t(rows)*c.ng);c.sub_min.resize(size_t(rows)*c.ng);
    mfq::bench::fill(c.q_packed,seed);mfq::bench::fill(c.sub_scale,seed);mfq::bench::fill(c.sub_min,seed);
    for(auto& v:c.sub_scale)v&=31;
    for(auto& v:c.sub_min)v&=15;
    c.neuron_scale_h.resize(rows);c.neuron_min_h.resize(rows);
    const auto scale=__float2half_rn(.000123f),minimum=__float2half_rn(.004731f);
    for(int row=0;row<rows;++row) {
        std::memcpy(&c.neuron_scale_h[row],&scale,2);
        std::memcpy(&c.neuron_min_h[row],&minimum,2);
    }
    return to_device_nint(c,true);
}
void run(const char* name,int rows,int width,int bits,int gs,bool grouped,int mode,
         const mfq::bench::Settings& settings) {
    const Device gpu{DeviceType::cuda,0};
    CudaExecutionContext execution;execution.config.diagnostic_nint_group=true;
    std::vector<float> values(width),gates(width);
    for(int i=0;i<width;++i){values[i]=std::sin(float(i+7)*.193f)*.31f;gates[i]=std::cos(float(i)*.13f);}
    auto x=tensor(values).reshape({1,width}).to(gpu,kFloat16);
    auto gate=tensor(gates).reshape({1,width}).to(gpu,kFloat16);
    const size_t bytes=size_t(rows)*((size_t(width)*bits+7)/8+2*((width+gs-1)/gs)+8);
    const size_t count=bytes<1024*1024 ? 4 : std::max<size_t>(2,(settings.working_bytes+bytes-1)/bytes);
    std::vector<NintWeight> weights;
    for(size_t i=0;i<count;++i) {
        auto w=weight(grouped?rows/2:rows,width,bits,gs,uint32_t(20261021+i));
        if(grouped) {
            QuantLinear a,b;a.nint=w;b.nint=weight(rows/2,width,bits,gs,uint32_t(20261077+i));
            w=make_quant_group(execution,{a,b}).nint.w;
        }
        weights.push_back(std::move(w));
    }
    const auto call=[&](size_t i) {
        return mode ? nint_matmul_input_mul(execution.profiler,weights[i],x,gate,mode)
                    : nint_matmul(execution.profiler,weights[i],x);
    };
    for(bool rotating:{false,true}) {
        const size_t sets=rotating?weights.size():1,calls=sets*4;
        const auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
        MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
        std::vector<Tensor> outputs;
        for(size_t i=0;i<calls;++i)outputs.push_back(call(i%sets));
        check(hipStreamSynchronize(stream.stream()));outputs.clear();
        graph.capture_begin();
        for(size_t i=0;i<calls;++i)outputs.push_back(call(i%sets));
        graph.capture_end();graph.replay();check(hipStreamSynchronize(stream.stream()));
        hipEvent_t start,stop;check(hipEventCreate(&start));check(hipEventCreate(&stop));
        const auto measure=[&](size_t repeats) {
            check(hipEventRecord(start,stream.stream()));
            for(size_t i=0;i<repeats;++i)graph.replay();
            check(hipEventRecord(stop,stream.stream()));check(hipEventSynchronize(stop));
            float ms;check(hipEventElapsedTime(&ms,start,stop));return double(ms);
        };
        size_t repeats=1;
        while(measure(repeats)<settings.target_ms)repeats*=2;
        std::vector<double> samples;
        for(int sample=0;sample<settings.samples;++sample)samples.push_back(measure(repeats)*1000/(calls*repeats));
        check(hipEventDestroy(start));check(hipEventDestroy(stop));
        std::cout<<"{\"name\":\""<<name<<"\",\"scope\":\"quantization_and_gemv_device\",\"cache\":\""
            <<(rotating?"rotating":"hot")<<"\",\"rows\":"<<rows<<",\"width\":"<<width
            <<",\"bits\":"<<bits<<",\"group_size\":"<<gs<<",\"working_sets\":"<<sets
            <<",\"canonical_weight_bytes\":"<<bytes<<",\"median_us\":"<<mfq::bench::median(samples)
            <<",\"samples_us\":";mfq::bench::values(samples);std::cout<<"}\n"<<std::flush;
    }
}
}
int main(int argc,char** argv) try {
    const auto settings=mfq::bench::settings(argc,argv);auto context=default_context(0);
    std::cout<<std::setprecision(9);
    run("gate_up",18432,2560,4,24,true,0,settings);
    run("gated_output",2560,4096,4,24,false,2,settings);
    run("down_nint6",2560,9216,6,24,false,0,settings);
    run("q8_projection",2560,2560,8,48,false,0,settings);
    run("q8_control",32,2560,8,48,false,0,settings);
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
