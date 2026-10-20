#pragma once
#include <algorithm>
#include <limits>

void check_contiguous_convert() {
    using namespace mfq::cuda;
    int checks=0,graphs=0;
    const auto exact=[](const Tensor& actual,const Tensor& expected) {
        auto a=actual.contiguous().to(kCPU),b=expected.contiguous().to(kCPU);
        require(a.scalar_type()==b.scalar_type() && a.sizes()==b.sizes() &&
            (!a.numel() || std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0),
            "contiguous conversion changed reference bytes");
    };
    const auto run=[](const Tensor& input,ScalarType destination,bool old) {
        set_test_environment("MFQ_DISABLE_NATIVE_CONTIGUOUS_CONVERT",old?"1":"0");
        return input.to(destination);
    };
    const std::vector<ScalarType> types={kBool,kUInt8,kInt8,kInt16,kInt32,kInt64,
        kFloat16,kBFloat16,kFloat32,kFloat64};
    // Exhaustive 16-bit floating encodings include signed zero, subnormals,
    // both infinities and every NaN payload. Integer/raw FP encodings also
    // exercise values that a small range of generated decimals would miss.
    for(const auto source:types) {
        auto host=empty({65538},TensorOptions{}.device(kCPU).dtype(source));
        auto* bytes=static_cast<uint8_t*>(host.data_ptr());
        for(int64_t i=0;i<host.nbytes();++i)bytes[i]=uint8_t(i*37+i/13);
        if(source==kFloat16 || source==kBFloat16) {
            auto* words=static_cast<uint16_t*>(host.data_ptr());
            for(int i=0;i<65538;++i)words[i]=uint16_t(i);
        }
        if(source==kBool)for(int64_t i=0;i<host.nbytes();++i)bytes[i]&=1;
        auto base=host.to(kCUDA);
        for(const auto destination:types)if(source!=destination) {
            for(int count:{0,1,257,65536}) {
                const auto input=base.narrow(0,1,count);
                auto expected=run(input,destination,true),actual=run(input,destination,false);
                exact(actual,expected);++checks;
            }
            auto input=base.narrow(0,0,65536).reshape({256,256}).transpose(0,1);
            exact(run(input,destination,false),run(input,destination,true));++checks;
            auto strided_destination=empty({256,257},base.options().dtype(destination)).narrow(1,0,256);
            set_test_environment("MFQ_DISABLE_NATIVE_CONTIGUOUS_CONVERT","0");
            strided_destination.copy_(input);
            exact(strided_destination,run(input,destination,true));++checks;
        }
    }
    for(const auto pair:std::vector<std::pair<ScalarType,ScalarType>>{
            {kFloat16,kFloat32},{kFloat32,kFloat16},{kBFloat16,kFloat32},{kFloat32,kBFloat16}}) {
        auto input=empty({65536},TensorOptions{}.device(kCUDA).dtype(pair.first));
        auto output=empty({65536},input.options().dtype(pair.second));
        set_test_environment("MFQ_DISABLE_NATIVE_CONTIGUOUS_CONVERT","0");
        Graph graph;graph.prepare_memory();output.copy_(input);
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
        graph.capture_begin();output.copy_(input);graph.capture_end();
        for(int step=0;step<3;++step) {
            input.copy_(arange(65536,TensorOptions{}.device(kCUDA).dtype(kFloat32)).remainder(4093)/(step+1));
            graph.replay();exact(output,run(input,pair.second,true));++graphs;
        }
    }
    // Compare full production-sized input conversions with preallocated output.
    for(const auto pair:std::vector<std::pair<ScalarType,ScalarType>>{
            {kFloat16,kFloat32},{kFloat32,kFloat16}}) {
        auto input=full({4048,10240},.37,TensorOptions{}.device(kCUDA).dtype(pair.first));
        auto output=empty(input.sizes(),input.options().dtype(pair.second));
        std::vector<float> times[2];
        for(int sample=0;sample<5;++sample)for(int order=0;order<2;++order) {
            const bool old=(sample+order)%2==0;
            set_test_environment("MFQ_DISABLE_NATIVE_CONTIGUOUS_CONVERT",old?"1":"0");
            output.copy_(input);MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
            cudaEvent_t begin,end;MFQ_NATIVE_CUDA_CHECK(cudaEventCreate(&begin));MFQ_NATIVE_CUDA_CHECK(cudaEventCreate(&end));
            MFQ_NATIVE_CUDA_CHECK(cudaEventRecord(begin,current_stream().stream()));
            for(int i=0;i<8;++i)output.copy_(input);
            MFQ_NATIVE_CUDA_CHECK(cudaEventRecord(end,current_stream().stream()));MFQ_NATIVE_CUDA_CHECK(cudaEventSynchronize(end));
            float ms=0;MFQ_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));times[old?0:1].push_back(ms/8);
            MFQ_NATIVE_CUDA_CHECK(cudaEventDestroy(begin));MFQ_NATIVE_CUDA_CHECK(cudaEventDestroy(end));
        }
        for(auto& values:times)std::sort(values.begin(),values.end());
        std::cout<<"contiguous_convert source="<<int(pair.first)<<" destination="<<int(pair.second)
            <<" elements="<<input.numel()<<" old_ms="<<times[0][2]<<" new_ms="<<times[1][2]<<'\n';
    }
    set_test_environment("MFQ_DISABLE_NATIVE_CONTIGUOUS_CONVERT","0");
    std::cout<<"contiguous_convert exact="<<checks<<" changing_graph="<<graphs<<" exhaustive_half_bf16=1 PASS\n";
}
