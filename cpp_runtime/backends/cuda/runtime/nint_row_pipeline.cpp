#include "nint_row_pipeline.h"
#include "mfq/nint_row_cache.h"
#include "mfq_cuda_quant_ops.h"
#include "mfq/host_parallel.h"
#include "mfq_cuda_context.h"
#include <atomic>
#include <future>
#include <map>
namespace tb=mfq_tensor_backend;
namespace {
struct Gathered {mfq::NintRowBatch batch;std::vector<int64_t> inverse;};
int64_t shape_rows(const std::vector<int64_t>& shape) {
    int64_t n=1;
    for(auto x:shape) {
        if(x<0 || (x && n>INT64_MAX/x))throw std::overflow_error("PLE output shape overflow");
        n*=x;
    }
    return n;
}
// Sources share immutable encoded rows, with source identity in every key.
std::shared_ptr<mfq::NintRowCache> row_cache() {
    static auto cache=std::make_shared<mfq::NintRowCache>(1u<<20);return cache;
}
}
struct NintRowStage::Impl {
    uint64_t source;
    int64_t rows;
    int width;
    std::size_t payload_bytes,host_bytes;
    std::vector<int64_t> shape;
    mfq::cuda::HostBuffer host;
    tb::Tensor packed,descriptors,inverse,flat,shaped;
    Impl(uint64_t identity,int64_t n,int w,std::size_t capacity,
        std::vector<int64_t> dimensions,const tb::Device& device)
        :source(identity),rows(n),width(w),payload_bytes(capacity),host_bytes(capacity+n*32),
         shape(std::move(dimensions)),host(host_bytes) {
        const MfqCudaGuard guard(device);
        auto options=tb::TensorOptions().device(device);
        packed=tb::zeros({static_cast<int64_t>(capacity)},options.dtype(tb::kUInt8));
        std::vector<int32_t> initial(static_cast<std::size_t>(n)*6,0);
        for(int64_t row=0;row<n;++row)initial[row*6+4]=w;
        descriptors=tb::tensor(initial).reshape({n,6}).to(device);
        inverse=tb::zeros({n},options.dtype(tb::kInt64));
        flat=tb::zeros({n,w},options.dtype(tb::kFloat16));
        auto out_shape=shape;out_shape.push_back(w);shaped=flat.reshape(out_shape);
        mfq::cuda::charge_tensor_host_bytes(host_bytes);
    }
    ~Impl(){mfq::cuda::tensor_host_bytes.fetch_sub(host_bytes);}
};
NintRowStage::NintRowStage(std::unique_ptr<Impl> impl):impl_(std::move(impl)){}
NintRowStage::~NintRowStage()=default;
tb::Tensor NintRowStage::output() const {return impl_->shaped;}
void NintRowStage::decode() {
    auto& s=*impl_;nint_selected_rows_into_cuda(s.packed,s.descriptors,s.inverse,s.flat);
}
struct NintRowPipeline::Impl {
    std::vector<std::shared_ptr<mfq::NintRows>> tables;
    std::shared_ptr<mfq::NintRowCache> cache=row_cache();
    uint64_t source_id=0;int64_t rows=0;int width=0;
    std::vector<int64_t> current_ids,current_shape;
    std::future<Gathered> pending;std::unique_ptr<Gathered> ready;
    explicit Impl(std::vector<std::shared_ptr<mfq::NintRows>> data):tables(std::move(data)) {
        static std::atomic<uint64_t> next{1};source_id=next.fetch_add(1);
        if(tables.empty() || !tables.front())throw std::invalid_argument("MFQ PLE needs row sources");
        rows=tables.front()->rows();width=tables.front()->width();
        for(const auto& table:tables)if(!table || table->rows()!=rows || table->width()!=width)
            throw std::invalid_argument("MFQ PLE shard dimensions differ");
        if(rows*static_cast<int64_t>(tables.size())>int64_t(UINT32_MAX))throw std::overflow_error("PLE logical row cache key overflow");
        if(source_id>UINT32_MAX)throw std::overflow_error("PLE source cache identity overflow");
    }
    Gathered gather(const std::vector<int64_t>& ids) {
        std::map<int64_t,std::shared_ptr<const mfq::NintRowBatch>> selected;
        std::vector<std::vector<int64_t>> misses(tables.size());
        for(const auto id:ids) {
            if(id<0 || id/rows>=static_cast<int64_t>(tables.size()))throw std::out_of_range("PLE logical row ID");
            if(selected.contains(id))continue;
            auto value=cache->find((source_id<<32)|static_cast<uint32_t>(id));selected.emplace(id,std::move(value));
            if(!selected.at(id))misses[id/rows].push_back(id%rows);
        }
        std::vector<mfq::NintRowBatch> fetched(tables.size());
        mfq::host_parallel_for(0,static_cast<int64_t>(tables.size()),1,mfq_get_num_threads(),[&](int64_t first,int64_t last) {
            for(auto shard=first;shard<last;++shard)if(!misses[shard].empty())
                tables[shard]->append_rows(misses[shard].data(),misses[shard].size(),fetched[shard],1);
        });
        for(std::size_t shard=0;shard<tables.size();++shard)for(std::size_t i=0;i<misses[shard].size();++i) {
            auto value=std::make_shared<mfq::NintRowBatch>();
            fetched[shard].copy_row(i,*value);
            const auto id=int64_t(shard)*rows+misses[shard][i];selected.at(id)=value;
            cache->insert((source_id<<32)|static_cast<uint32_t>(id),std::move(value));
        }
        Gathered out;std::map<int64_t,int64_t> locations;
        for(const auto& [id,value]:selected){locations.emplace(id,out.batch.rows());out.batch.append_batch(*value);}
        for(auto id:ids)out.inverse.push_back(locations.at(id));return out;
    }
};
NintRowPipeline::NintRowPipeline(std::vector<std::shared_ptr<mfq::NintRows>> data):impl_(std::make_unique<Impl>(std::move(data))){}
NintRowPipeline::~NintRowPipeline()=default;
void NintRowPipeline::issue(const std::vector<int64_t>& ids,const std::vector<int64_t>& shape) {
    auto& s=*impl_;if(s.current_ids==ids && s.current_shape==shape && (s.ready || s.pending.valid()))return;
    if(s.pending.valid())s.ready=std::make_unique<Gathered>(s.pending.get());
    s.current_ids=ids;s.current_shape=shape;s.ready.reset();
    s.pending=std::async(std::launch::async,[&s,ids]{return s.gather(ids);});
}
tb::Tensor NintRowPipeline::collect(const std::vector<int64_t>& ids,const std::vector<int64_t>& shape,const tb::Device& device) {
    auto& s=*impl_;issue(ids,shape);
    if(s.pending.valid())s.ready=std::make_unique<Gathered>(s.pending.get());
    const auto& result=*s.ready;auto out_shape=shape;out_shape.push_back(s.width);
    const auto n=shape_rows(shape);
    if(n!=static_cast<int64_t>(ids.size()) || !device.is_cuda())throw std::invalid_argument("PLE pipeline input shape/device mismatch");
    const MfqCudaGuard guard(device);
    if(ids.empty())return tb::empty(out_shape,tb::TensorOptions().device(device).dtype(tb::kFloat16));
    result.batch.validate();auto packed=tb::tensor(result.batch.packed()).to(device);
    std::vector<int32_t> words(result.batch.descriptors().size());std::memcpy(words.data(),result.batch.descriptors().data(),words.size()*4);
    auto descriptors=tb::tensor(words).reshape({static_cast<int64_t>(result.batch.rows()),6}).to(device);
    return nint_selected_rows_cuda(packed,descriptors,s.width).index_select(0,tb::tensor(result.inverse).to(device)).reshape(out_shape);
}
std::shared_ptr<NintRowStage> NintRowPipeline::make_stage(const std::vector<int64_t>& shape,const tb::Device& device) {
    const auto& s=*impl_;const auto n=shape_rows(shape);
    if(!device.is_cuda() || n<=0 || n>INT_MAX/6 || uint64_t(n)*s.width>UINT32_MAX)
        throw std::invalid_argument("PLE graph stage shape/device exceeds bounds");
    std::size_t row_bytes=0;
    for(const auto& table:s.tables)row_bytes=std::max(row_bytes,table->selected_row_nbytes_bound());
    if(row_bytes>std::size_t(INT_MAX)/n)throw std::overflow_error("PLE graph payload exceeds bounds");
    const auto capacity=(row_bytes*n+7)&~std::size_t(7);
    return std::shared_ptr<NintRowStage>(new NintRowStage(std::make_unique<NintRowStage::Impl>(
        s.source_id,n,s.width,capacity,shape,device)));
}
void NintRowPipeline::upload(NintRowStage& stage,const std::vector<int64_t>& ids,const std::vector<int64_t>& shape) {
    auto& s=*impl_;auto& target=*stage.impl_;
    if(target.source!=s.source_id || target.shape!=shape || int64_t(ids.size())!=target.rows)
        throw std::invalid_argument("PLE graph upload differs from its prepared stage");
    issue(ids,shape);
    if(s.pending.valid())s.ready=std::make_unique<Gathered>(s.pending.get());
    const auto& result=*s.ready;result.batch.validate();
    if(result.batch.packed_nbytes()>target.payload_bytes || result.batch.rows()>std::size_t(target.rows) ||
       result.inverse.size()!=ids.size())throw std::overflow_error("PLE graph upload exceeds its prepared stage");
    auto* host=static_cast<uint8_t*>(target.host.data());
    const auto descriptor_bytes=result.batch.descriptors().size()*sizeof(uint32_t);
    auto* descriptors=host+target.payload_bytes;
    auto* inverse=descriptors+target.rows*6*sizeof(uint32_t);
    std::memcpy(host,result.batch.packed().data(),result.batch.packed_nbytes());
    std::memcpy(descriptors,result.batch.descriptors().data(),descriptor_bytes);
    std::memcpy(inverse,result.inverse.data(),result.inverse.size()*sizeof(int64_t));
    const MfqCudaGuard guard(target.packed.device());const auto stream=mfq_current_cuda_stream();
    MFQ_CUDA_CHECK(cudaMemcpyAsync(target.packed.data_ptr(),host,result.batch.packed_nbytes(),cudaMemcpyHostToDevice,stream));
    MFQ_CUDA_CHECK(cudaMemcpyAsync(target.descriptors.data_ptr(),descriptors,descriptor_bytes,cudaMemcpyHostToDevice,stream));
    MFQ_CUDA_CHECK(cudaMemcpyAsync(target.inverse.data_ptr(),inverse,result.inverse.size()*sizeof(int64_t),cudaMemcpyHostToDevice,stream));
}
