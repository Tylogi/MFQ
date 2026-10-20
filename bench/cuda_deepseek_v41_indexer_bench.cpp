#include "mfq/kernels/cuda/deepseek_v4_attention.h"
#include "mfq_cuda_context.h"
#include <cmath>
#include <iostream>

namespace tb=mfq_tensor_backend;
template<class F> double elapsed(F function,int repetitions) {
    for(int i=0;i<4;++i)function();
    mfq::cuda::Event begin(cudaEventDefault),end(cudaEventDefault);
    auto stream=mfq_current_cuda_stream();begin.record(stream);
    for(int i=0;i<repetitions;++i)function();
    end.record(stream);end.synchronize();
    float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
    return ms/repetitions;
}
int main() try {
    const auto fp32=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    for(int pool:{4096,131072}) for(int tokens:{1,128}) {
        const int count=std::min(pool,16384);
        auto query=tb::randn({1,tokens,32,128},fp32).to(tb::kFloat16).contiguous();
        auto keys=tb::randn({1,pool,128},fp32).to(tb::kFloat16).contiguous();
        auto weights=tb::randn({1,tokens,32},fp32).to(tb::kFloat16).contiguous();
        std::vector<int64_t> selected(tokens*count);
        for(int t=0;t<tokens;++t)for(int c=0;c<count;++c)
            selected[t*count+c]=(int64_t(c)*17+t*31)%pool;
        auto candidates=tb::tensor(selected).reshape({1,tokens,count}).to(tb::kCUDA).contiguous();
        auto full=[&]{return dsv4_indexer_scores_cuda(query,keys,weights,pool-tokens,1)
                                .gather(-1,candidates);};
        auto indexed=[&]{return dsv4_indexer_candidate_scores_cuda(query,keys,weights,candidates,
                                                                   pool-tokens,1);};
        if(!full().equal(indexed()))throw std::runtime_error("C4A benchmark scoring mismatch");
        const int repetitions=tokens==1?100:10;
        const auto full_ms=elapsed(full,repetitions),indexed_ms=elapsed(indexed,repetitions);
        std::cout<<"{\"pool\":"<<pool<<",\"tokens\":"<<tokens<<",\"candidates\":"<<count
                 <<",\"previous_ms\":"<<full_ms<<",\"indexed_ms\":"<<indexed_ms
                 <<",\"speedup\":"<<full_ms/indexed_ms<<"}"<<std::endl;
    }
    return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
