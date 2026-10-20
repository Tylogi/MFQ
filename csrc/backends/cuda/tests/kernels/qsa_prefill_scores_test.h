#pragma once
#include "native/env.h"

void qsa_live_pool_write_case() {
    int cases=0;const auto stream=mfq_current_cuda_stream();
    for(int rows:{1,2,6,32})for(int pool:{4,8})for(int width:{32,128,256})
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int axes:{1,3}) {
        constexpr int capacity=1057;
        auto index=data({1,capacity,width+8},13,.4f).to(dtype).narrow(2,0,width);
        auto weight=data({width},19,.1f).to(tb::kFloat32);
        auto cp=tb::zeros({rows},index.options().dtype(tb::kInt64));
        auto positions=tb::zeros({axes,1,rows},index.options().dtype(axes==1?tb::kInt64:tb::kInt32));
        const int dim=std::min(width,64),pairs=dim/2;
        RotaryEmbedding rotary(dim,capacity+256,10000.,axes==3?std::vector<int64_t>{pairs/2,pairs/4,pairs/4}:std::vector<int64_t>{},axes==3,true);
        Tensor outputs[2]={tb::zeros({1,(capacity+pool-1)/pool,width},index.options()),
            tb::zeros({1,(capacity+pool-1)/pool,width},index.options())};Graph graphs[2];
        const auto compute=[&](int mode) {
            if(mode) {
                if(!rotary.pool_normalized_into(index,weight,positions,cp,outputs[1],pool,capacity,1e-6))
                    throw std::runtime_error("fused pool write did not dispatch");
            }else {
                auto p=mfq_qwen4_exp::live_index_pool(index,positions,cp,pool,capacity);
                auto pooled=rotary.forward_normalized(p[0].unsqueeze(2),weight,p[2],1e-6).squeeze(1);
                outputs[0].index_copy_(1,p[1],pooled);
            }
        };
        for(int mode=0;mode<2;++mode) {
            {GraphWarmupScope warmup(default_context(mfq_current_cuda_device()),stream);compute(mode);}
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
            graphs[mode].capture_begin_shared();compute(mode);graphs[mode].capture_end();
        }
        for(int start:{0,3,4,1019,1056}) {
            std::vector<int64_t> cache(rows),pos(axes*rows);
            for(int row=0;row<rows;++row) {
                cache[row]=std::min(start,capacity-rows)+row;
                for(int axis=0;axis<axes;++axis)pos[axis*rows+row]=cache[row]+axis*31-7;
            }
            cp.copy_(tb::tensor(cache).to(tb::kCUDA));
            positions.copy_(tb::tensor(pos).reshape(positions.sizes()).to(positions.scalar_type()).to(tb::kCUDA));
            outputs[0].zero_();outputs[1].zero_();graphs[0].replay();graphs[1].replay();
            equal(outputs[1],outputs[0],"QSA fused pooled cache",0,0);++cases;
        }
        if((rows==1 || rows==6) && pool==4 && width==128 && dtype==tb::kFloat16 && axes==3)for(int mode:{0,1,0}) {
            Graph graph;graph.capture_begin_shared();for(int i=0;i<64;++i)compute(mode);graph.capture_end();
            for(int i=0;i<16;++i)graph.replay();
            cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
            MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));for(int i=0;i<128;++i)graph.replay();
            MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
            std::cout<<"qsa_pool_write rows="<<rows<<" fused="<<mode<<" us="<<ms*1000/(128*64)<<'\n';
        }
    }
    std::cout<<"qsa_pool_write_cases="<<cases<<" exact cache, changing graph, partial pools, MRoPE PASS\n";
}

void qsa_live_pool_case() {
    int cases=0;
    const auto stream=mfq_current_cuda_stream();
    for(int rows:{1,6})for(int pool:{4,8})for(int width:{32,128})
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int axes:{1,3}) {
        constexpr int capacity=1057;
        auto index=data({1,capacity,width},13,.4f).to(dtype);
        auto weight=data({width},19,.1f).to(tb::kFloat32);
        auto cp=tb::zeros({rows},index.options().dtype(tb::kInt64));
        auto positions=tb::zeros({axes,1,rows},index.options().dtype(axes==1?tb::kInt64:tb::kInt32));
        const int dim=std::min(width,64),pairs=dim/2;
        RotaryEmbedding rotary(dim,capacity+256,10000.,axes==3?std::vector<int64_t>{pairs/2,pairs/4,pairs/4}:std::vector<int64_t>{},axes==3,true);
        Tensor averages[2],blocks[2],block_pos[2],outputs[2];Graph graphs[2];
        const auto compute=[&](int mode) {
            if(mode) {
                auto p=mfq_qwen4_exp::live_index_pool(index,positions,cp,pool,capacity);
                averages[mode]=p[0];blocks[mode]=p[1];block_pos[mode]=p[2];
                outputs[mode]=rotary.forward_normalized(p[0].unsqueeze(2),weight,p[2],1e-6).squeeze(1);
            }else {
                blocks[mode]=(cp/pool).to(tb::kInt64);
                auto selected=(blocks[mode].unsqueeze(-1)*pool+tb::arange(pool,cp.options())).clamp(0,capacity-1);
                averages[mode]=index.index_select(1,selected.reshape({-1})).reshape({1,rows,pool,width}).to(tb::kFloat32).mean(2).to(dtype);
                block_pos[mode]=blocks[mode]*pool+(positions-cp);
                auto normalized=attention_ops::rms_norm(averages[mode],weight+1,1e-6);
                outputs[mode]=rotary.forward(normalized.unsqueeze(1),block_pos[mode]).squeeze(1);
            }
        };
        for(int mode=0;mode<2;++mode) {
            {GraphWarmupScope warmup(default_context(mfq_current_cuda_device()),stream);compute(mode);}
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
            averages[mode]={};blocks[mode]={};block_pos[mode]={};outputs[mode]={};
            graphs[mode].capture_begin_shared();compute(mode);graphs[mode].capture_end();
        }
        for(int start:{0,3,4,1019,1056}) {
            std::vector<int64_t> cache(rows),pos(axes*rows);
            for(int row=0;row<rows;++row) {
                cache[row]=std::min(start,capacity-rows)+row;
                for(int axis=0;axis<axes;++axis)pos[axis*rows+row]=cache[row]+axis*31-7;
            }
            cp.copy_(tb::tensor(cache).to(tb::kCUDA));
            positions.copy_(tb::tensor(pos).reshape(positions.sizes()).to(positions.scalar_type()).to(tb::kCUDA));
            graphs[0].replay();graphs[1].replay();
            equal(averages[1],averages[0],"QSA live pooled mean",0,0);
            equal(blocks[1],blocks[0],"QSA live block IDs",0,0);
            equal(block_pos[1],block_pos[0],"QSA live block positions",0,0);
            equal(outputs[1],outputs[0],"QSA live pool norm/rope",dtype==tb::kFloat16?2e-3f:2e-6f,2e-6f);
            ++cases;
        }
        if(rows==1 && pool==4 && width==128 && dtype==tb::kFloat16 && axes==3)for(int mode:{0,1,0}) {
            for(int i=0;i<16;++i)graphs[mode].replay();
            cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
            MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));
            for(int i=0;i<256;++i)graphs[mode].replay();
            MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
            std::cout<<"qsa_live_pool fused="<<mode<<" us="<<ms*1000/256<<'\n';
        }
    }
    std::cout<<"qsa_live_pool_cases="<<cases<<" changing graph, partial pools, MRoPE PASS\n";
}

void qsa_live_topk_tile_case() {
    struct Setting {
        std::string old;
        Setting(){if(const auto* p=mfq::cuda::env_lookup("MFQ_QSA_TOPK_LIVE_TILE"))old=p;}
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("MFQ_QSA_TOPK_LIVE_TILE",value);
#else
            if(*value)setenv("MFQ_QSA_TOPK_LIVE_TILE",value,1);else unsetenv("MFQ_QSA_TOPK_LIVE_TILE");
#endif
        }
        ~Setting(){set(old.c_str());}
    } setting;
    const auto stream=mfq_current_cuda_stream();
    int cases=0;
    for(int pools:{2048,32768})for(int rows:{1,6})for(int count:{128,512,1024}) {
        auto scores=tb::zeros({1,rows,pools},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32));
        auto positions=tb::zeros({rows},scores.options().dtype(tb::kInt64));
        Tensor outputs[2];Graph graphs[2];
        for(int mode=0;mode<2;++mode) {
            Setting::set(mode?"1":"0");
            const auto compute=[&]{outputs[mode]=mfq_qwen4_exp::live_topk_blocks(scores,positions,4,count);};
            {GraphWarmupScope warmup(default_context(mfq_current_cuda_device()),stream);compute();}
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));outputs[mode]={};
            graphs[mode].capture_begin_shared();compute();graphs[mode].capture_end();
        }
        for(int pattern=0;pattern<3;++pattern) {
            std::vector<float> values(rows*pools);
            for(std::size_t i=0;i<values.size();++i)
                values[i]=pattern==0?0.f:pattern==1?float((i*17)%7):std::sin(float(i)*.319f);
            scores.copy_(tb::tensor(values).reshape(scores.sizes()).to(tb::kCUDA));
            for(int live:{0,127,128,511,512,513,1023,1024,1025,2047,2048,2049,4095,4096,4097,8193,32768}) {
                std::vector<int64_t> pos(rows);
                for(int row=0;row<rows;++row)pos[row]=int64_t(std::min(pools,live+row))*4-1;
                positions.copy_(tb::tensor(pos).to(tb::kCUDA));
                graphs[0].replay();graphs[1].replay();
                auto a=outputs[0].cpu().contiguous(),b=outputs[1].cpu().contiguous();
                if(a.sizes()!=b.sizes() || std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))
                    throw std::runtime_error("live QSA tile changed ordered IDs at live="+std::to_string(live));
                ++cases;
            }
        }
        if(rows==1 && pools==32768 && count==512)for(int live:{600,1500,3000,8193}) {
            positions.copy_(tb::tensor(std::vector<int64_t>{live*4-1}).to(tb::kCUDA));
            for(int mode:{0,1,0}) {
                for(int i=0;i<16;++i)graphs[mode].replay();
                cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
                MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));
                for(int i=0;i<256;++i)graphs[mode].replay();
                MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
                float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
                MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
                std::cout<<"qsa_live_tile visible="<<live<<" compact="<<mode<<" us="<<ms*1000/256<<'\n';
            }
        }
    }
    std::cout<<"qsa_live_tile_cases="<<cases<<" changing graph, tied scores, ordered IDs exact PASS\n";
}

void qsa_live_scores_case() {
    qsa_live_pool_case();
    qsa_live_topk_tile_case();
    struct Setting {
        std::string old;
        Setting(){if(const auto* p=mfq::cuda::env_lookup("MFQ_QSA_LIVE_SCORES"))old=p;}
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("MFQ_QSA_LIVE_SCORES",value);
#else
            if(*value)setenv("MFQ_QSA_LIVE_SCORES",value,1);else unsetenv("MFQ_QSA_LIVE_SCORES");
#endif
        }
        ~Setting(){set(old.c_str());}
    } setting;
    int cases=0;
    for(int rows:{1,4,8})for(int pools:{2048,32768})for(int width:{32,128})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        constexpr int heads=8,pool=4;
        auto make=[](int64_t count,int salt) {
            std::vector<float> values(count);
            for(int64_t i=0;i<count;++i) {
                uint32_t hash=uint32_t(i)+uint32_t(salt)*7919u;
                hash^=hash>>16;hash*=0x7feb352du;hash^=hash>>15;hash*=0x846ca68bu;hash^=hash>>16;
                values[i]=(float(hash&65535u)/32768.f-1.f)*.13f;
            }
            return tb::tensor(values).to(tb::kCUDA);
        };
        auto query=make(rows*heads*width,17).reshape({1,heads,rows,width}).to(dtype).permute({0,2,1,3});
        auto pooled=make(pools*width,31).reshape({1,pools,width}).to(tb::kFloat16);
        for(int64_t start:{int64_t(0),int64_t(4093),int64_t(pools*pool-rows)}) {
            auto positions=tb::arange(start,start+rows,pooled.options().dtype(tb::kInt64));
            Setting::set("0");auto expected=mfq_qwen4_exp::live_block_scores(query,pooled,positions,pool);
            Setting::set("1");auto actual=mfq_qwen4_exp::live_block_scores(query,pooled,positions,pool);
            equal(actual,expected,"live QSA scores and visibility",2e-6f,2e-5f);
            auto old_indices=std::get<1>(tb::topk(expected,128,-1,true,true)).cpu().contiguous();
            auto new_indices=std::get<1>(tb::topk(actual,128,-1,true,true)).cpu().contiguous();
            for(int row=0;row<rows;++row) {
                auto* a=old_indices.data_ptr<int64_t>()+row*128;
                auto* b=new_indices.data_ptr<int64_t>()+row*128;
                std::sort(a,a+128);std::sort(b,b+128);
                for(int i=0;i<128;++i)if(a[i]!=b[i])throw std::runtime_error("live QSA selected a different block set");
            }
            for(int count:{128,512,1024}) {
                auto reference=std::get<1>(tb::topk(expected,count,-1,true,false)).cpu().contiguous();
                auto selected=mfq_qwen4_exp::live_topk_blocks(actual,positions,pool,count).cpu().contiguous();
                for(int row=0;row<rows;++row) {
                    const auto visible=(start+row+1)/pool;
                    std::vector<int64_t> a,b;
                    for(int i=0;i<count;++i) {
                        const auto old=reference.data_ptr<int64_t>()[row*count+i],
                            now=selected.data_ptr<int64_t>()[row*count+i];
                        if(old>=0 && old<visible)a.push_back(old);
                        if(now>=0 && now<visible)b.push_back(now);
                    }
                    std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
                    if(a!=b)throw std::runtime_error("live QSA top-k changed the visible block set");
                }
            }
            if(rows==1 && pools==32768 && width==128 && dtype==tb::kFloat16 && start>0) {
                float timing[2]{};
                for(int mode=0;mode<2;++mode) {
                    cudaEvent_t first,last;MFQ_CUDA_CHECK(cudaEventCreate(&first));MFQ_CUDA_CHECK(cudaEventCreate(&last));
                    MFQ_CUDA_CHECK(cudaEventRecord(first,mfq_current_cuda_stream()));
                    for(int i=0;i<32;++i) {
                        if(mode)(void)mfq_qwen4_exp::live_topk_blocks(actual,positions,pool,512);
                        else (void)tb::topk(actual,512,-1,true,false);
                    }
                    MFQ_CUDA_CHECK(cudaEventRecord(last,mfq_current_cuda_stream()));MFQ_CUDA_CHECK(cudaEventSynchronize(last));
                    MFQ_CUDA_CHECK(cudaEventElapsedTime(timing+mode,first,last));
                    MFQ_CUDA_CHECK(cudaEventDestroy(first));MFQ_CUDA_CHECK(cudaEventDestroy(last));timing[mode]/=32;
                }
                std::cout<<"qsa_live_topk visible="<<(start+1)/pool<<" original_ms="<<timing[0]<<" live_ms="<<timing[1]<<'\n';
            }
            ++cases;
        }
    }
    std::cout<<"qsa_live_score_cases="<<cases<<" score/selection oracle PASS\n";
}

void qsa_prefill_scores_case() {
    struct Setting {
        std::string old;
        Setting(){if(const auto* p=mfq::cuda::env_lookup("MFQ_QSA_PREFILL_SCORE_GEMM"))old=p;}
        static void set(const char* value) {
            if(*value)mfq::cuda::env_set("MFQ_QSA_PREFILL_SCORE_GEMM",value);else mfq::cuda::env_clear("MFQ_QSA_PREFILL_SCORE_GEMM");
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
