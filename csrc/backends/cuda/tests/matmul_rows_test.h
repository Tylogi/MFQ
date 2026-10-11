#pragma once

void verify_matmul_rows() {
    using namespace mfq::cuda;
    const Device device{DeviceType::cuda,0};
    int cases=0;
    for(auto dtype:{kFloat16,kBFloat16,kFloat32})for(int columns:{48,640,2560,6144})
    for(int rows:{8192,8201}) {
        constexpr int width=2560,tile=1024;
        std::vector<float> xs(size_t(rows)*width),ws(size_t(columns)*width);
        for(size_t i=0;i<xs.size();++i)xs[i]=std::sin(float(i)*.017f)*.13f;
        for(size_t i=0;i<ws.size();++i)ws[i]=std::cos(float(i)*.013f)*.09f;
        auto x=tensor(xs).reshape({rows,width}).to(device).to(dtype);
        auto w=tensor(ws).reshape({columns,width}).to(device).to(dtype).transpose(0,1);
        const auto serial=[&] {
            std::vector<Tensor> parts;
            for(int begin=0;begin<rows;begin+=tile)
                parts.push_back(matmul(x.narrow(0,begin,std::min(tile,rows-begin)),w));
            return cat(parts,0);
        };
        auto expected=serial().to(kCPU).contiguous();
        for(int streams:{1,2,4}) {
            Tensor result;
            {
                const MatmulRowsScope scope(tile,streams);
                result=matmul(x,w);
            }
            auto checked=result.to(kCPU).contiguous();
            require(std::memcmp(checked.data_ptr(),expected.data_ptr(),expected.nbytes())==0,
                    "parallel row GEMM changed original output bits");
            ++cases;
        }
        if(rows==8192 && dtype==kFloat16 && (columns==48 || columns==6144)) {
            for(int streams:{0,2,4}) {
                const auto run=[&] {
                    if(!streams)return serial();
                    const MatmulRowsScope scope(tile,streams);return matmul(x,w);
                };
                auto result=run();result=run();
                MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
                cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);
                cudaEventRecord(start,current_stream().stream());
                for(int i=0;i<12;++i)result=run();
                cudaEventRecord(stop,current_stream().stream());cudaEventSynchronize(stop);
                float ms=0;cudaEventElapsedTime(&ms,start,stop);cudaEventDestroy(start);cudaEventDestroy(stop);
                std::cout<<"matmul_rows_pair rows="<<rows<<" N="<<columns<<" K="<<width
                    <<" tile="<<tile<<" streams="<<streams<<" milliseconds="<<ms/12<<std::endl;
            }
        }
    }
    std::cout<<"matmul_rows_exact_cases="<<cases<<" PASS"<<std::endl;
}

void verify_parallel_row_chains() {
    using namespace mfq::cuda;
    const Device device{DeviceType::cuda,0};
    constexpr int width=256,columns=64,tile=256;
    int cases=0;
    for(auto dtype:{kFloat16,kBFloat16,kFloat32})for(int rows:{2048,2059}) {
        std::vector<float> xs(size_t(rows)*width),gs(size_t(width)*columns),us(gs.size()),ds(gs.size());
        for(size_t i=0;i<xs.size();++i)xs[i]=std::sin(float(i)*.017f)*.13f;
        for(size_t i=0;i<gs.size();++i) {
            gs[i]=std::cos(float(i)*.013f)*.025f;
            us[i]=std::sin(float(i)*.011f)*.021f;
            ds[i]=std::cos(float(i)*.019f)*.023f;
        }
        auto x=tensor(xs).reshape({rows,width}).to(device).to(dtype);
        auto g=tensor(gs).reshape({columns,width}).to(device).to(dtype).transpose(0,1);
        auto u=tensor(us).reshape({columns,width}).to(device).to(dtype).transpose(0,1);
        auto d=tensor(ds).reshape({width,columns}).to(device).to(dtype).transpose(0,1);
        const auto project=[&](const Tensor& part){return matmul(matmul(part,g)*matmul(part,u),d);};
        std::vector<Tensor> expected_parts;
        for(int begin=0;begin<rows;begin+=tile)
            expected_parts.push_back(project(x.narrow(0,begin,std::min(tile,rows-begin))));
        auto expected=cat(expected_parts,0).to(kCPU).contiguous();
        const auto main=current_stream().stream();
        for(int streams:{1,2,4}) {
            auto output=empty({rows,width},x.options());
            output.narrow(0,0,tile).copy_(project(x.narrow(0,0,tile)));
            submit_parallel_rows(0,tile,rows,tile,streams,[&](int64_t begin,int64_t count) {
                output.narrow(0,begin,count).copy_(project(x.narrow(0,begin,count)));
            });
            require(current_stream().stream()==main,"parallel row chain changed caller stream");
            auto checked=output.to(kCPU).contiguous();
            require(std::memcmp(checked.data_ptr(),expected.data_ptr(),expected.nbytes())==0,
                    "parallel row pipeline changed original output bits");
            ++cases;
        }
        bool caught=false;
        try {
            submit_parallel_rows(0,tile,rows,tile,2,[&](int64_t begin,int64_t count) {
                auto part=project(x.narrow(0,begin,count));
                if(begin==tile*3)throw std::runtime_error("test submission failure");
            });
        } catch(const std::runtime_error&){caught=true;}
        require(caught && current_stream().stream()==main,"failed row chain did not restore caller state");
        auto after=project(x.narrow(0,0,tile)).to(kCPU).contiguous();
        require(std::memcmp(after.data_ptr(),expected.data_ptr(),after.nbytes())==0,
                "failed row pipeline did not restore cuBLAS handle");
    }
    std::cout<<"parallel_row_chain_exact_cases="<<cases<<" exception_restore_cases=6 PASS"<<std::endl;
}
