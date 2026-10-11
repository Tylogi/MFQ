#pragma once
#include <cstdint>
#include <functional>

namespace mfq::cuda {
// Retain the original GEMM row count while evaluating independent chunks on
// several streams. The scope is used only by whole-layer prefill projections.
class MatmulRowsScope final {
public:
    explicit MatmulRowsScope(std::int64_t rows,int streams);
    ~MatmulRowsScope() noexcept;
    MatmulRowsScope(const MatmulRowsScope&)=delete;
    MatmulRowsScope& operator=(const MatmulRowsScope&)=delete;
private:
    std::int64_t previous_rows_;
    int previous_streams_;
};
// Submit complete independent row pipelines on worker streams. Callers prepare
// shared weights and allocate their destination on the main stream first.
void submit_parallel_rows(int device,std::int64_t begin,std::int64_t rows,
    std::int64_t chunk,int streams,const std::function<void(std::int64_t,std::int64_t)>& submit);
}
