#pragma once

void qsa_prefill_scores_case() {
    struct Setting {
        std::string old;
        Setting(){if(const auto* p=std::getenv("MFQ_QSA_PREFILL_SCORE_GEMM"))old=p;}
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("MFQ_QSA_PREFILL_SCORE_GEMM",value);
#else
            if(*value)setenv("MFQ_QSA_PREFILL_SCORE_GEMM",value,1);else unsetenv("MFQ_QSA_PREFILL_SCORE_GEMM");
#endif
        }
        ~Setting(){set(old.c_str());}
    } setting;
    int cases=0;
    for(int rows:{33,1024,2048})for(int pools:{64,2048}) {
        constexpr int heads=4,width=128;
        auto make=[](int64_t count,int salt) {
            std::vector<float> values(count);
            for(int64_t i=0;i<count;++i) {
                uint32_t hash=uint32_t(i)+uint32_t(salt)*7919u;
                hash^=hash>>16;hash*=0x7feb352du;hash^=hash>>15;hash*=0x846ca68bu;hash^=hash>>16;
                values[i]=(float(hash&65535u)/32768.f-1.f)*.13f;
            }
            return tb::tensor(values).to(tb::kCUDA);
        };
        auto query=make(rows*heads*width,17).reshape({1,rows,heads,width});
        auto pooled=make(pools*width,31).reshape({1,pools,width}).to(tb::kFloat16);
        Setting::set("0");auto expected=mfq_qwen4_exp::block_scores(query,pooled);
        Setting::set("1");auto actual=mfq_qwen4_exp::block_scores(query,pooled);
        equal(actual,expected,"QSA prefill matrix scores",1e-7f,1e-5f);
        auto old_indices=std::get<1>(tb::topk(expected,std::min(128,pools),-1,true,true)).cpu().contiguous();
        auto new_indices=std::get<1>(tb::topk(actual,std::min(128,pools),-1,true,true)).cpu().contiguous();
        int64_t differences=0,set_differences=0;
        for(int64_t i=0;i<old_indices.numel();++i)differences+=
            old_indices.data_ptr<int64_t>()[i]!=new_indices.data_ptr<int64_t>()[i];
        const auto selected=std::min(128,pools);
        for(int row=0;row<rows;++row) {
            const auto* old_row=old_indices.data_ptr<int64_t>()+int64_t(row)*selected;
            const auto* new_row=new_indices.data_ptr<int64_t>()+int64_t(row)*selected;
            std::vector<int64_t> old_set(old_row,old_row+selected),new_set(new_row,new_row+selected);
            std::sort(old_set.begin(),old_set.end());std::sort(new_set.begin(),new_set.end());
            for(int i=0;i<selected;++i)set_differences+=old_set[i]!=new_set[i];
        }
        auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
        double error=0,signal=0;
        for(int64_t i=0;i<a.numel();++i) {
            const double difference=double(a.data_ptr<float>()[i])-b.data_ptr<float>()[i];
            error+=difference*difference;signal+=double(b.data_ptr<float>()[i])*b.data_ptr<float>()[i];
        }
        float times[2]{};
        for(int mode=0;mode<2;++mode) {
            Setting::set(mode?"1":"0");
            for(int i=0;i<2;++i)(void)mfq_qwen4_exp::block_scores(query,pooled);
            cudaEvent_t first,last;MFQ_CUDA_CHECK(cudaEventCreate(&first));MFQ_CUDA_CHECK(cudaEventCreate(&last));
            MFQ_CUDA_CHECK(cudaEventRecord(first,mfq_current_cuda_stream()));
            for(int i=0;i<8;++i)(void)mfq_qwen4_exp::block_scores(query,pooled);
            MFQ_CUDA_CHECK(cudaEventRecord(last,mfq_current_cuda_stream()));MFQ_CUDA_CHECK(cudaEventSynchronize(last));
            MFQ_CUDA_CHECK(cudaEventElapsedTime(&times[mode],first,last));
            MFQ_CUDA_CHECK(cudaEventDestroy(first));MFQ_CUDA_CHECK(cudaEventDestroy(last));
            times[mode]/=8;
        }
        std::cout<<"qsa_prefill_score_pair rows="<<rows<<" pools="<<pools<<" original_ms="<<times[0]
            <<" candidate_ms="<<times[1]<<" relative_rms="<<std::sqrt(error/std::max(1e-30,signal))
            <<" ordered_topk_differences="<<differences<<" selected_set_differences="<<set_differences<<std::endl;
        ++cases;
    }
    std::cout<<"qsa_prefill_score_cases="<<cases<<" FP32 score oracle PASS"<<std::endl;
}
