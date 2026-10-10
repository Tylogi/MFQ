// Preserve router and shared-expert arithmetic while regrouping routed experts.
static thread_local int64_t qwen_prefill_reference_rows=0;
struct QwenPrefillReferenceRowsScope {
    int64_t previous;
    explicit QwenPrefillReferenceRowsScope(int64_t rows):previous(qwen_prefill_reference_rows) {
        qwen_prefill_reference_rows=rows;
    }
    ~QwenPrefillReferenceRowsScope(){qwen_prefill_reference_rows=previous;}
};
template<class Projection>
static Tensor qwen_prefill_preserve_rows(const Tensor& source,Projection project) {
    const auto rows=qwen_prefill_reference_rows;
    if(rows<=0 || source.size(0)<=rows)return project(source);
    Tensor output;
    for(int64_t begin=0;begin<source.size(0);begin+=rows) {
        const auto count=std::min<int64_t>(rows,source.size(0)-begin);
        auto part=project(source.narrow(0,begin,count));
        if(!output.defined()) {
            auto shape=part.sizes().vec();shape[0]=source.size(0);
            output=tb::empty(shape,part.options());
        }
        output.narrow(0,begin,count).copy_(part);
    }
    return output;
}
