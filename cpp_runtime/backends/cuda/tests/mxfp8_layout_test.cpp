#include "mx.h"
#include "mfq_cuda_quant_ops.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq/runtime.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tb=mfq_tensor_backend;
namespace {
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
float decode(uint8_t raw) {
    const int exponent=(raw>>3)&15,mantissa=raw&7;
    const float magnitude=exponent?std::ldexp(1+float(mantissa)/8,exponent-7):std::ldexp(float(mantissa),-9);
    return raw&128?-magnitude:magnitude;
}
void verify(const tb::Tensor& value,const std::vector<float>& expected) {
    const auto actual=value.to(tb::kCPU).to(tb::kFloat32).contiguous();
    require(actual.numel()==expected.size(),"MXFP8 output shape differs");
    for(std::size_t i=0;i<expected.size();++i)
        if(!std::isfinite(actual.data_ptr<float>()[i]) ||
            std::abs(actual.data_ptr<float>()[i]-expected[i])>2e-4f+1e-3f*std::abs(expected[i]))
            throw std::runtime_error("MXFP8 independent oracle mismatch at "+std::to_string(i)+
                " actual="+std::to_string(actual.data_ptr<float>()[i])+" expected="+std::to_string(expected[i]));
}
template <typename Project>
void verify_token_rows(const tb::Tensor& value,const tb::Tensor& input,Project project) {
    const auto actual=value.cpu().contiguous();
    const auto row_bytes=std::size_t(actual.size(1))*actual.element_size();
    for(int64_t row=0;row<input.size(0);++row) {
        const auto single=project(input.narrow(0,row,1)).cpu().contiguous();
        if(std::memcmp(static_cast<const std::byte*>(actual.data_ptr())+row*row_bytes,
                single.data_ptr(),row_bytes))
            throw std::runtime_error("MXFP8 multi-token summation differs from single-token projection rows="+
                std::to_string(input.size(0))+" width="+std::to_string(input.size(-1)));
    }
}
void check(int block_rows,int block_columns,int outputs,int width,int groups=1,
        bool shifted=false,bool shifted_input=false) {
    std::vector<uint8_t> blob{'M','X','T','1',1,8,0,0};
    const int scale_rows=(outputs+block_rows-1)/block_rows,scale_columns=width/block_columns;
    for(uint64_t d:{uint64_t(outputs),uint64_t(width),uint64_t(outputs),uint64_t(width),
            uint64_t(scale_rows),uint64_t(scale_columns)}) {
        const auto at=blob.size();blob.resize(at+8);std::memcpy(blob.data()+at,&d,8);
    }
    for(int i=0;i<outputs*width;++i) {
        const auto raw=uint8_t((i*37+11)%240);blob.push_back((raw&127)==127?uint8_t(raw-1):raw);
    }
    for(int i=0;i<scale_rows*scale_columns;++i)blob.push_back(uint8_t(119+(i*7)%5));
    auto cpu=unpack_mxfp8(blob);
    require(cpu.block_rows==block_rows && cpu.block_columns==block_columns,"MXFP8 scale block was discarded");
    auto weight=to_device_mxfp8(cpu,true);
    if(shifted) {
        auto backing=tb::empty({weight.values.numel()+1},weight.values.options());
        auto values=backing.narrow(0,1,weight.values.numel()).reshape(weight.values.sizes().vec());
        values.copy_(weight.values);weight.values=std::move(values);
    }
    std::vector<float> dense(outputs*width);
    for(int n=0;n<outputs;++n)for(int k=0;k<width;++k)
        dense[std::size_t(n)*width+k]=float(mfq_half(decode(cpu.values[std::size_t(n)*width+k])*
            std::ldexp(1.0f,int(cpu.scales[(n/block_rows)*scale_columns+k/block_columns])-127)));
    verify(mxfp8_dequant_cuda(weight.values,weight.scales),dense);
    verify(mxfp8_cpu_reference(weight),dense);
    CudaProfiler profiler;
    for(int m:{1,2,3,4,5,6,7,8,17,31,32,33,63,64,65,129}) {
        std::vector<float> input(std::size_t(m)*groups*width),expected(std::size_t(m)*outputs);
        for(std::size_t i=0;i<input.size();++i)input[i]=float(int(i*13%17)-8)/32;
        const int per=outputs/groups;
        for(int row=0;row<m;++row)for(int n=0;n<outputs;++n) {
            double sum=0;
            for(int k=0;k<width;++k)sum+=double(input[(std::size_t(row)*groups+n/per)*width+k])*dense[std::size_t(n)*width+k];
            expected[std::size_t(row)*outputs+n]=float(sum);
        }
        auto x=tb::tensor(input).reshape({m,groups,width}).to(tb::kFloat16).to(tb::kCUDA);
        if(shifted_input) {
            auto backing=tb::empty({x.numel()+1},x.options());
            auto shifted_x=backing.narrow(0,1,x.numel()).reshape(x.sizes().vec());
            shifted_x.copy_(x);x=std::move(shifted_x);
        }
        if(groups==1) {
            x=x.reshape({m,width});
            verify(mxfp8_matmul(profiler,weight,x),expected);
            const auto precise=mxfp8_matmul_f32(profiler,weight,x);
            verify(precise,expected);
            if(m<=8)verify_token_rows(precise,x,[&](const auto& row){return mxfp8_matmul_f32(profiler,weight,row);});
            verify(mxfp8_matmul(profiler,to_device_mxfp8(cpu,false),x.to(tb::kCPU)),expected);
        } else {
            verify(mxfp8_groupwise_matmul(profiler,weight,x,groups),expected);
            const auto precise=mxfp8_groupwise_matmul_f32(profiler,weight,x,groups);
            verify(precise,expected);
            if(m<=8)verify_token_rows(precise,x,[&](const auto& row){return mxfp8_groupwise_matmul_f32(profiler,weight,row,groups);});
            if(m>8)verify(mxfp8_groupwise_matmul(profiler,to_device_mxfp8(cpu,false),x.to(tb::kCPU),groups),expected);
        }
    }
    auto input_slice=slice_mxfp8_cpu(cpu,TensorParallelAxis::Input,block_columns,width);
    require(input_slice.block_columns==block_columns && input_slice.block_rows==block_rows &&
        input_slice.scales.size()==std::size_t(scale_rows)*(scale_columns-1),"MXFP8 input slice scale geometry changed");
    const int first=block_rows==1?3:block_rows;
    auto output_slice=slice_mxfp8_cpu(cpu,TensorParallelAxis::Output,first,outputs);
    require(output_slice.scales.size()==std::size_t((outputs-first+block_rows-1)/block_rows)*scale_columns,
        "MXFP8 output slice scale geometry changed");
    std::cout<<"MXFP8 "<<block_rows<<'x'<<block_columns<<" outputs="<<outputs<<" width="<<width<<" groups="<<groups<<" PASS\n";
}
}
void check_mxfp4_routes(int tokens,int outputs,int width,bool routed) {
    constexpr int experts=5,pool=3,routes=3;
    const std::vector<int32_t> local{1,-1,2,0,-1};
    const int pairs=tokens*routes;
    std::vector<int32_t> ids(pairs),compact,bounds{0},tile_bounds{0},tile_experts;
    for(int i=0;i<pairs;++i)ids[i]=(i*i+3*i+i/7)%experts;
    for(int expert=0;expert<experts;++expert) {
        for(int i=0;i<pairs;++i)if(ids[i]==expert)compact.push_back(i);
        const int count=int(compact.size())-bounds.back();
        bounds.push_back(int(compact.size()));
        tile_bounds.push_back(tile_bounds.back()+(count+7)/8);
        for(int i=0;i<(count+7)/8;++i)tile_experts.push_back(expert);
    }
    std::vector<uint8_t> values(std::size_t(pool)*outputs*width/2);
    std::vector<uint8_t> scales(std::size_t(pool)*outputs*width/32);
    for(std::size_t i=0;i<values.size();++i)values[i]=uint8_t((i*157+23)%256);
    for(std::size_t i=0;i<scales.size();++i)scales[i]=uint8_t(119+(i*7)%9);
    std::vector<float> input(std::size_t(routed?pairs:tokens)*width),expected(std::size_t(pairs)*outputs,-7);
    for(std::size_t i=0;i<input.size();++i)input[i]=float(int(i*13%29)-14)/32;
    constexpr float magnitude[]{0,.5f,1,1.5f,2,3,4,6};
    for(int pair=0;pair<pairs;++pair)if(local[ids[pair]]>=0)for(int n=0;n<outputs;++n) {
        const int neuron=local[ids[pair]]*outputs+n;
        double sum=0;
        for(int k=0;k<width;++k) {
            const auto code=(values[std::size_t(neuron)*(width/2)+k/2]>>((k&1)*4))&15;
            const float weight=(code&8?-magnitude[code&7]:magnitude[code&7])*
                std::ldexp(1.0f,int(scales[std::size_t(neuron)*(width/32)+k/32])-127);
            sum+=double(input[std::size_t(routed?pair:pair/routes)*width+k])*weight;
        }
        expected[std::size_t(pair)*outputs+n]=float(sum);
    }
    const auto u8=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    auto packed=tb::tensor(values).reshape({pool*outputs,width/2}).to(tb::kCUDA);
    auto scale=tb::tensor(scales).reshape({pool*outputs,width/32}).to(tb::kCUDA);
    auto x=tb::tensor(input).to(tb::kFloat16).to(tb::kCUDA);
    x=routed?x.reshape({tokens,routes,width}):x.reshape({tokens,width});
    auto route_ids=tb::tensor(ids).reshape({tokens,routes}).to(tb::kCUDA);
    auto locals=tb::tensor(local).to(tb::kCUDA);
    auto positions=tb::tensor(compact).to(tb::kCUDA),starts=tb::tensor(bounds).to(tb::kCUDA);
    auto tiles=tb::tensor(tile_bounds).to(tb::kCUDA),owners=tb::tensor(tile_experts).to(tb::kCUDA);
    auto empty=tb::empty({0},u8.dtype(tb::kInt32));
    for(bool grouped:{false,true}) {
        auto output=tb::full({tokens,routes,outputs},-7,u8.dtype(tb::kFloat16));
        auto result=mxfp4_moe_grouped_matmul_pool_f16_cuda(packed,scale,x,route_ids,locals,
            experts,pool,outputs,width,output,grouped?positions:empty,starts,tiles,owners);
        verify(result,expected);
    }
    std::cout<<"MXFP4 routes tokens="<<tokens<<" outputs="<<outputs<<" width="<<width
             <<" routed="<<routed<<" PASS\n";
}

void benchmark_mxfp4_prefill() {
    constexpr int tokens=4080,routes=6,experts=384,pool=16,pairs=tokens*routes;
    std::vector<int32_t> ids(pairs),local(experts,-1),compact,bounds{0},tile_bounds{0},tile_experts;
    for(int i=0;i<pairs;++i)ids[i]=i%experts;
    for(int e=0;e<experts;++e) {
        if(e<pool)local[e]=e;
        for(int i=e;i<pairs;i+=experts)compact.push_back(i);
        const auto count=int(compact.size())-bounds.back();bounds.push_back(int(compact.size()));
        tile_bounds.push_back(tile_bounds.back()+(count+7)/8);
        for(int i=0;i<(count+7)/8;++i)tile_experts.push_back(e);
    }
    auto gpu=[](const auto& values){return tb::tensor(values).to(tb::kCUDA);};
    auto route_ids=gpu(ids).reshape({tokens,routes}),locals=gpu(local),positions=gpu(compact);
    auto starts=gpu(bounds),tiles=gpu(tile_bounds),owners=gpu(tile_experts);
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    for(const auto shape:{std::array<int,2>{5120,2304},std::array<int,2>{2304,5120}}) {
        const auto [width,outputs]=shape;
        auto packed=tb::full({pool*outputs,width/2},0x22,options);
        auto scales=tb::full({pool*outputs,width/32},127,options);
        auto input=tb::full({tokens,width},.03125,options.dtype(tb::kFloat16));
        auto output=tb::full({tokens,routes,outputs},-7,options.dtype(tb::kFloat16));
        const auto launch=[&] {
            mxfp4_moe_grouped_matmul_pool_f16_cuda(packed,scales,input,route_ids,locals,
                experts,pool,outputs,width,output,positions,starts,tiles,owners);
        };
        for(int i=0;i<2;++i)launch();
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        cudaEvent_t start,stop;MFQ_CUDA_CHECK(cudaEventCreate(&start));MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        MFQ_CUDA_CHECK(cudaEventRecord(start,mfq_current_cuda_stream()));
        constexpr int iterations=20;
        for(int i=0;i<iterations;++i)launch();
        MFQ_CUDA_CHECK(cudaEventRecord(stop,mfq_current_cuda_stream()));MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
        auto sample=output.select(0,0).to(tb::kCPU).to(tb::kFloat32).contiguous();
        for(int r=0;r<routes;++r)for(int n=0;n<outputs;++n)
            require(sample.data_ptr<float>()[r*outputs+n]==width*.03125f,"MXFP4 prefill independent oracle differs");
        std::cout<<"mxfp4_prefill rows="<<tokens<<" width="<<width<<" outputs="<<outputs
            <<" ms="<<ms/iterations<<" iterations="<<iterations<<" warmup=2\n";
        MFQ_CUDA_CHECK(cudaEventDestroy(start));MFQ_CUDA_CHECK(cudaEventDestroy(stop));
    }
}

void benchmark_prefill(bool only_ffn=false) {
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    for(const auto shape:{std::array<int,3>{4080,1536,32768},std::array<int,3>{4080,5120,2304}}) {
        const auto [rows,width,outputs]=shape;
        if(only_ffn && outputs==32768)continue;
        auto values=tb::full({outputs,width},0x38,options);
        auto scales=tb::full({outputs/32,width/32},127,options);
        auto input=tb::full({rows,width},.03125,options.dtype(tb::kFloat16));
        cudaEvent_t start,stop;MFQ_CUDA_CHECK(cudaEventCreate(&start));MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        MFQ_CUDA_CHECK(cudaEventRecord(start,mfq_current_cuda_stream()));
        auto result=mxfp8_matmul_f16_cuda(values,scales,input);
        MFQ_CUDA_CHECK(cudaEventRecord(stop,mfq_current_cuda_stream()));MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
        auto sample=result.select(0,0).to(tb::kCPU).to(tb::kFloat32).contiguous();
        for(int n=0;n<outputs;++n)require(sample.data_ptr<float>()[n]==width*.03125f,"MXFP8 large prefill oracle differs");
        std::cout<<"mxfp8_prefill rows="<<rows<<" width="<<width<<" outputs="<<outputs<<" ms="<<ms<<'\n';
        MFQ_CUDA_CHECK(cudaEventDestroy(start));MFQ_CUDA_CHECK(cudaEventDestroy(stop));
    }
}
enum class Mxfp8BenchPath { PackedSimt, PackedTensorCore };
enum class Mxfp8BenchBatch { Decode, Draft, Prefill };
void benchmark_native_mxfp8(Mxfp8BenchPath path,Mxfp8BenchBatch batch=Mxfp8BenchBatch::Decode) {
    const bool prefill=batch==Mxfp8BenchBatch::Prefill;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    for(const auto shape:{std::array<int,3>{1280,32768,1}, {5120,2304,1},
            {2304,5120,1}, {5120,512,1}, {5120,1280,1},
            {4096,8192,8}, {8192,5120,1}, {15360,5120,1}}) {
        const auto [width,outputs,groups]=shape;
        if(path==Mxfp8BenchPath::PackedTensorCore && groups!=1 && !prefill)continue;
        std::array<float,256> decoded{};
        for(int code=0;code<256;++code)decoded[code]=decode(uint8_t(code));
        std::vector<uint8_t> bytes(std::size_t(outputs)*width);
        std::vector<float> sums(outputs);
        for(int n=0;n<outputs;++n) {
            double sum=0;
            for(int k=0;k<width;++k) {
                const auto i=std::size_t(n)*width+k;
                const auto code=uint8_t((i*37+n*13)%127|(((i*29+n)&1)<<7));
                bytes[i]=code;sum+=decoded[code]*.03125;
            }
            sums[n]=float(sum);
        }
        auto values=tb::tensor(bytes).reshape({outputs,width}).to(tb::kCUDA);
        auto scales=tb::full({(outputs+31)/32,width/32},127,options);
        Mxfp8Weight weight;weight.values=values;weight.scales=scales;
        weight.out=outputs;weight.neuron_len=width;
        CudaProfiler profiler;
        const std::vector<int> row_counts=prefill?std::vector<int>{128,1024,4080}:
            batch==Mxfp8BenchBatch::Draft?std::vector<int>{2,3,4,5,6,8}:std::vector<int>{1,2,4,8};
        const int repeats=prefill?3:40;
        for(const int rows:row_counts) {
            auto input=tb::full({rows,groups,width},.03125,options.dtype(tb::kFloat16));
            const auto run=[&] {
                if(path==Mxfp8BenchPath::PackedTensorCore) {
                    if(groups!=1)return mxfp8_groupwise_matmul(profiler,weight,input,groups);
                    return mxfp8_matmul_f16_cuda(values,scales,input.reshape({rows,width}));
                }
                return groups==1 ? mxfp8_small_m_cuda(values,scales,input.reshape({rows,width}))
                    : mxfp8_groupwise_small_m_cuda(values,scales,input,groups);
            };
            for(int i=0;i<(prefill?1:5);++i)run();
            cudaEvent_t start,stop;
            MFQ_CUDA_CHECK(cudaEventCreate(&start));MFQ_CUDA_CHECK(cudaEventCreate(&stop));
            MFQ_CUDA_CHECK(cudaEventRecord(start,mfq_current_cuda_stream()));
            for(int i=0;i<repeats;++i)run();
            MFQ_CUDA_CHECK(cudaEventRecord(stop,mfq_current_cuda_stream()));
            MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
            float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
            if(prefill) {
                const auto result=run();
                verify(result.select(0,0),sums);verify(result.select(0,rows-1),sums);
            }else {
                std::vector<float> expected;
                expected.reserve(std::size_t(rows)*outputs);
                for(int row=0;row<rows;++row)expected.insert(expected.end(),sums.begin(),sums.end());
                verify(run(),expected);
            }
            std::cout<<(prefill?"mxfp8_prefill":"mxfp8_decode")<<" rows="<<rows<<" width="<<width<<" outputs="<<outputs
                <<" groups="<<groups<<" scale_block=32 path="
                <<(path==Mxfp8BenchPath::PackedTensorCore?"tensor_core":"packed_simt")
                <<" ms="<<ms/repeats<<std::endl;
            MFQ_CUDA_CHECK(cudaEventDestroy(start));MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        }
    }
}
int main(int argc,char** argv) try {
    if(argc==2 && std::string(argv[1])=="--bench-decode") {benchmark_native_mxfp8(Mxfp8BenchPath::PackedSimt);return 0;}
    if(argc==2 && std::string(argv[1])=="--bench-decode-tc") {benchmark_native_mxfp8(Mxfp8BenchPath::PackedTensorCore);return 0;}
    if(argc==2 && std::string(argv[1])=="--bench-native-prefill") {benchmark_native_mxfp8(Mxfp8BenchPath::PackedTensorCore,Mxfp8BenchBatch::Prefill);return 0;}
    if(argc==2 && std::string(argv[1])=="--bench-mtp") {benchmark_native_mxfp8(Mxfp8BenchPath::PackedSimt,Mxfp8BenchBatch::Draft);return 0;}
    if(argc==2 && std::string(argv[1])=="--bench-mxfp4") {benchmark_mxfp4_prefill();return 0;}
    if(argc==2 && std::string(argv[1])=="--mxfp4-only") {
        for(bool routed:{false,true})for(const auto shape:{std::array<int,3>{1,19,32},
                std::array<int,3>{7,65,96},std::array<int,3>{33,65,96},std::array<int,3>{129,57,512}})
            check_mxfp4_routes(shape[0],shape[1],shape[2],routed);
        return 0;
    }
    if(argc>=2 && std::string(argv[1])=="--bench-prefill") {benchmark_prefill(argc==3);return 0;}
    check(128,128,137,256);
    check(32,32,65,96);
    check(128,128,137,256,1,true);
    check(32,32,65,96,1,true);
    check(1,32,65,96,1,true);
    check(1,32,65,96);
    check(128,128,137,2048);
    check(32,32,65,160);
    check(128,128,137,256,1,false,true);
    check(32,32,96,96,3);
    check(1,32,96,96,3);
    check(128,128,384,256,3);
    check(32,32,288,160,3,true);
    check_mxfp4_routes(129,65,96,false);
    check_mxfp4_routes(129,65,96,true);
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
