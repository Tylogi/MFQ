#include "mfq_cuda_nvq1_decode.h"
#include "mfq_cuda_quant_ops.h"
#include "vq.h"
#include "timing.h"
#include <array>
#include <cstdlib>
#include <cstring>

namespace tb=mfq_tensor_backend;
constexpr int rows=17,local_experts=3,routes=10,experts=4;
static int projection_checks=0;
static int cpu_checks=0;
static double cpu_worst=0;
static void flag(const char* name,bool on) {
#ifdef _WIN32
    _putenv_s(name,on?"1":"0");
#else
    setenv(name,on?"1":"0",1);
#endif
}
static void put(std::vector<uint8_t>& data,int64_t bit,int width,uint32_t value) {
    for(int i=0;i<width;++i)if((value>>i)&1u)data[(bit+i)/8]|=uint8_t(1u<<((bit+i)&7));
}
static std::vector<tb::Tensor> projection_outputs(NvqWeight& w) {
    CudaExecutionContext execution;std::vector<tb::Tensor> outputs;
    const auto input=[&](int batch) {
        return tb::tensor(format_bench::input(batch,int(w.neuron_len)))
            .reshape({batch,w.neuron_len}).to(tb::kCUDA,tb::kFloat16);
    };
    for(int batch:{1,3,8,16,64})
        outputs.push_back(nvq_matmul(execution.profiler,w,input(batch)).to(tb::kCPU));
    for(int batch:{16,64}) {
        const auto x=input(batch);auto& workspace=w.workspace(batch);
        outputs.push_back(nvq_mmq_ws_cuda(w.indices_packed,w.aux_packed,w.sub_scale_packed,
            w.neuron_scale,w.codebook,x,w.neuron_len,w.gs,w.sub_bits,w.kernel_format,w.sign_mode,
            workspace.qx,workspace.xscale).to(tb::kCPU));
    }
    outputs.push_back(nvq_gemm_f16_cuda(w.indices_packed,w.aux_packed,w.sub_scale_packed,
        w.neuron_scale,w.codebook,input(16),w.neuron_len,w.gs,w.sub_bits,w.kernel_format,w.sign_mode).to(tb::kCPU));
    outputs.push_back(nvq_matmul_multi2(execution.profiler,w,w,input(3)).to(tb::kCPU));
    outputs.push_back(nvq_matmul_swiglu(execution.profiler,w,w,input(1)).to(tb::kCPU));
    outputs.push_back(nvq_embedding(w,tb::tensor(std::vector<int64_t>{0,17,50}).to(tb::kCUDA)).to(tb::kCPU));
    return outputs;
}
static tb::Tensor cpu_projection(NvqWeight w) {
    for(auto* field:{&w.indices_packed,&w.aux_packed,&w.sub_scale_packed,&w.neuron_scale,&w.codebook})
        *field=field->to(tb::kCPU).contiguous();
    w.workspaces.clear();
    CudaExecutionContext execution;
    auto x=tb::tensor(format_bench::input(3,int(w.neuron_len))).reshape({3,w.neuron_len});
    return nvq_matmul(execution.profiler,w,x).to(tb::kFloat32);
}
static void check_case(bool small,int width,int tokens,bool down,bool records,bool integer_delta,int& count) {
    const int groups=(width+23)/24,vectors=(width+7)/8,bits=small?9:11,state_bits=small?4:3;
    const int entries=small?512:2048,all_rows=rows*local_experts,input_rows=tokens*(down?routes:1);
    std::vector<uint8_t> index((int64_t(all_rows)*vectors*bits+7)/8),
        auxiliary((all_rows*groups+7)/8),states((all_rows*groups*state_bits+7)/8);
    std::vector<int8_t> book((small?2:1)*entries*8);
    std::vector<float> anchors(all_rows);
    for(int i=0;i<int(book.size());++i)book[i]=int8_t((i*7+i/8*11+i/entries)%3-1);
    for(int row=0;row<all_rows;++row) {
        anchors[row]=float(row%7+1)*.000731f;
        for(int v=0;v<vectors;++v)put(index,(int64_t(row)*vectors+v)*bits,bits,(row*17+v*23)%entries);
        for(int g=0;g<groups;++g) {
            put(states,(int64_t(row)*groups+g)*state_bits,state_bits,(row*13+g*7)&((1u<<state_bits)-1));
            put(auxiliary,int64_t(row)*groups+g,1,(row*11+g*7)&1);
        }
    }
    NvqWeight w;w.kernel_format=w.format=small?8:1;w.gs=24;w.sub_bits=state_bits;
    w.out=all_rows;w.ng=groups;w.neuron_len=width;
    w.indices_packed=tb::tensor(index).to(tb::kCUDA);w.aux_packed=tb::tensor(auxiliary).to(tb::kCUDA);
    w.sub_scale_packed=tb::tensor(states).to(tb::kCUDA);
    w.codebook=tb::tensor(book).reshape({small?1024:2048,8}).to(tb::kCUDA);
    w.neuron_scale=tb::tensor(anchors).to(tb::kCUDA);
    const auto dequant=[&] {
        return nvq_dequant_cuda(w.indices_packed,w.aux_packed,w.sub_scale_packed,w.neuron_scale,w.codebook,
            w.neuron_len,w.gs,w.sub_bits,w.kernel_format,0).to(tb::kCPU);
    };
    auto canonical=dequant();
    const bool check_projections=records && !integer_delta && tokens==1 && !down;
    auto expected_projections=check_projections?projection_outputs(w):std::vector<tb::Tensor>{};
    auto expected_cpu=check_projections?cpu_projection(w):tb::Tensor{};
    flag("MFQ_NVQ1_GROUP_RECORDS",records);flag("MFQ_NVQ1_INTEGER_DELTA",integer_delta);
    prepare_nvq_decode_records(w);prepare_nvq_integer_codebook(w);
    auto converted=dequant();
    if(std::memcmp(canonical.data_ptr(),converted.data_ptr(),canonical.numel()*canonical.element_size()))
        throw std::runtime_error("NVQ1 record conversion changed dequantized weight bits");
    if(check_projections) {
        auto actual_cpu=cpu_projection(w);double error=0,norm=0;
        for(int64_t i=0;i<actual_cpu.numel();++i) {
            const double a=expected_cpu.data_ptr<float>()[i],b=actual_cpu.data_ptr<float>()[i];
            if(!std::isfinite(b))throw std::runtime_error("nonfinite CPU NVQ1 record result");
            error+=(a-b)*(a-b);norm+=a*a;
        }
        const double relative=std::sqrt(error/std::max(norm,1e-30));
        if(relative>2e-4)throw std::runtime_error("CPU NVQ1 record decode changed projection result");
        cpu_worst=std::max(cpu_worst,relative);++cpu_checks;
        auto actual=projection_outputs(w);
        for(size_t i=0;i<actual.size();++i) {
            const auto& a=expected_projections[i];const auto& b=actual[i];
            if(a.sizes()!=b.sizes() || std::memcmp(a.data_ptr(),b.data_ptr(),a.numel()*a.element_size()))
                throw std::runtime_error("NVQ1 record conversion changed matrix/fused/embedding output bits: path="+std::to_string(i));
            ++projection_checks;
        }
    }
    Nvq1DecodeView view;
    const std::array<tb::Tensor,7> fields={w.indices_packed,w.aux_packed,w.sub_scale_packed,w.neuron_scale,
        w.codebook,w.decode_records,w.integer_codebook};
    for(int i=0;i<7;++i)if(fields[i].defined())view.pointers[i]=reinterpret_cast<int64_t>(fields[i].data_ptr());
    view.sizes[0]=w.indices_packed.numel();view.sizes[1]=w.aux_packed.numel();view.sizes[2]=w.sub_scale_packed.numel();
    view.sizes[3]=records?w.decode_records.numel():0;
    view.groups=groups;view.vectors=vectors;view.format=int(w.kernel_format);view.local_experts=local_experts;
    auto x=tb::zeros(down?std::vector<int64_t>{tokens,routes,width}:std::vector<int64_t>{tokens,width},
        w.codebook.options().dtype(tb::kFloat16));
    auto ids=tb::zeros({tokens,routes},x.options().dtype(tb::kInt32));
    auto map=tb::zeros({experts},ids.options());
    auto qx=tb::empty({input_rows,groups*24},x.options().dtype(tb::kInt8));
    auto xs=tb::empty({input_rows,groups},x.options().dtype(tb::kFloat32));
    auto output=tb::zeros({tokens,routes,rows},x.options());
    const auto run=[&] {
        output.zero_();
        if(!nvq1_try_decode_cuda(view,x,ids,map,qx,xs,output,experts,rows,false))
            throw std::runtime_error("NVQ1 decode experiment declined supported oracle case");
        format_bench::check(cudaGetLastError());
    };
    MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);run();
    format_bench::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
    graph.capture_begin();run();graph.capture_end();
    for(int step=0;step<3;++step) {
        std::vector<float> values(input_rows*width);
        for(int i=0;i<int(values.size());++i)values[i]=step==1?0.f:float((i*37+step*11)%255-127)*.001231f;
        x.copy_(tb::tensor(values).reshape(x.sizes()).to(tb::kCUDA,tb::kFloat16));
        auto rounded=x.to(tb::kCPU,tb::kFloat32);
        std::vector<int8_t> codes(input_rows*groups*24);
        std::vector<float> scales(input_rows*groups);
        for(int m=0;m<input_rows;++m)for(int g=0;g<groups;++g) {
            float maximum=0;
            for(int e=0;e<24 && g*24+e<width;++e)maximum=std::max(maximum,std::abs(rounded.data_ptr<float>()[m*width+g*24+e]));
            const float scale=maximum>0?maximum/127.f:1.f;scales[m*groups+g]=scale;
            for(int e=0;e<24;++e)codes[(m*groups+g)*24+e]=g*24+e<width?
                int8_t(std::max(-127,std::min(127,int(std::round(rounded.data_ptr<float>()[m*width+g*24+e]/scale))))):0;
        }
        std::vector<int32_t> selected(tokens*routes),local{0,1,2,-1};
        if(step==1)local={2,-1,0,-1};
        for(int i=0;i<int(selected.size());++i)selected[i]=step==2?(i%3==0?-1:i%3==1?experts:2):i%experts;
        ids.copy_(tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA));map.copy_(tb::tensor(local).to(tb::kCUDA));
        graph.replay();auto actual=output.to(tb::kCPU,tb::kFloat32);
        for(int pair=0;pair<tokens*routes;++pair)for(int row=0;row<rows;++row) {
            const int expert=selected[pair],l=unsigned(expert)<unsigned(experts)?local[expert]:-1;
            double expected=0,magnitude=0;
            if(l>=0)for(int g=0;g<groups;++g) {
                const int neuron=l*rows+row,negative=(neuron*11+g*7)&1;
                const int state=(neuron*13+g*7)&((1u<<state_bits)-1);
                const int input=down?pair:pair/routes;
                int dot=0,sum=0;
                for(int e=0;e<24;++e) {
                    const int column=g*24+e,q=codes[(input*groups+g)*24+e];
                    const int index_value=(neuron*17+(column/8)*23)%entries;
                    const int v=book[((small?negative:0)*entries+index_value)*8+column%8];
                    dot+=v*q;sum+=q;
                }
                const double value=double(anchors[neuron])*state*scales[input*groups+g]*
                    (dot+(small?5./32.:1./8.)*(negative?-1:1)*sum);
                expected+=value;magnitude+=std::abs(value);
            }
            const double value=actual.data_ptr<float>()[int64_t(pair)*rows+row];
            if(!std::isfinite(value) || (l<0 && value!=0) ||
               std::abs(value-expected)>8e-4*std::abs(expected)+6e-5*magnitude+2e-7)
                throw std::runtime_error("NVQ1 routed record/integer FP64 oracle mismatch");
        }
        ++count;
    }
}
int main()try {
    int devices=0;const auto status=cudaGetDeviceCount(&devices);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && !devices))return 77;
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    auto context=mfq::cuda::default_context(0);auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    int count=0;
    const bool canonical_only=std::getenv("MFQ_TEST_NVQ1_CANONICAL_ONLY")!=nullptr;
    for(bool small:{false,true})for(int width:{1,25,640,2560})for(int tokens:{1,3,8})for(bool down:{false,true})
        for(auto layout:{std::pair{false,false},std::pair{true,false},std::pair{false,true},std::pair{true,true}}) {
            if(canonical_only && (layout.first || layout.second))continue;
            check_case(small,width,tokens,down,layout.first,layout.second,count);
        }
    std::cout<<"NVQ1 routed FP64 oracle: "<<count
        <<" cases, record/integer combinations, tails, zero inputs, both deltas, duplicate/masked/changing routes, graph replay PASS\n";
    if(projection_checks)std::cout<<"NVQ1 record layout: "<<projection_checks
        <<" raw/record bitexact dense, int8 MMA, FP16 tiles, multi2, SwiGLU, embedding checks PASS\n";
    if(cpu_checks)std::cout<<"NVQ1 CPU record fallback: "<<cpu_checks<<" checks, max_relative_l2="<<cpu_worst<<" PASS\n";
    return 0;
}catch(const std::exception& error){std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
