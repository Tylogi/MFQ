#include "moe_quant_range_source.h"
#include "quant_linear.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tb=mfq_tensor_backend;

namespace {
std::size_t expert_fields_bytes(const MixedMoePool& pool) {
    std::vector<tb::Tensor> fields;
    if (pool.family==MixedMoeFamily::Nint)
        fields={pool.nint.q_packed,pool.nint.row_q_bits,pool.nint.row_q_bit_offsets,
            pool.nint.sub_scale,pool.nint.sub_min,pool.nint.neuron_scale,pool.nint.neuron_min};
    else fields={pool.nvq.indices_packed,pool.nvq.aux_packed,pool.nvq.sub_scale_packed,pool.nvq.neuron_scale};
    std::size_t bytes=0;
    for (const auto& field:fields) bytes+=field.numel()*field.element_size();
    return bytes;
}
}

MoeQuantRangeSource::MoeQuantRangeSource(std::shared_ptr<mfq::MfeQuantExpertStore> store)
    :store_(std::move(store)),metadata_(std::make_shared<MixedMoeRuntime>()) {
    if (!store_) throw std::invalid_argument("missing quantized expert range source");
    metadata_->n_experts=store_->num_experts();
    metadata_->out_per_expert=store_->out_per_expert();
    metadata_->neuron_len=store_->neuron_len();
    q_strides_.assign(store_->pool_count(),0);
    field_bytes_.assign(store_->pool_count(),0);
    gpu_resident_=std::make_unique<std::atomic_bool[]>(store_->num_experts());
    for (int expert=0; expert<store_->num_experts(); ++expert) gpu_resident_[expert].store(false);
    for (std::size_t index=0; index<store_->pool_count(); ++index) {
        const auto& ids=store_->pool_expert_ids(index);
        if (store_->expert_dtype(ids.front())=="NINT") {
            std::uint64_t maximum=0;
            for (int expert:ids) maximum=std::max(maximum,store_->expert_values_bits(expert));
            const auto bytes=(maximum+7)/8+8;
            if (bytes>std::numeric_limits<std::int64_t>::max()) throw std::overflow_error("NINT expert stride overflow");
            q_strides_[index]=static_cast<std::int64_t>(bytes);
        }
        auto pool=read_expert(ids.front());
        field_bytes_[index]=expert_fields_bytes(pool);
        pool.local_experts=static_cast<int>(ids.size());
        std::vector<std::int32_t> local(store_->num_experts(),-1);
        for (std::size_t j=0; j<ids.size(); ++j) local[ids[j]]=static_cast<std::int32_t>(j);
        pool.expert_local=tb::tensor(local,tb::TensorOptions().dtype(tb::kInt32));
        metadata_->pools.push_back(std::move(pool));
    }
}

MixedMoePool MoeQuantRangeSource::read_expert(int expert) const {
    const auto encoded=store_->read_expert(expert);
    const auto index=store_->expert_pool(expert);
    MixedMoePool pool;
    pool.local_experts=1;
    if (encoded.dtype=="NINT") {
        pool.family=MixedMoeFamily::Nint;
        pool.nint=to_device_mfe_nint(unpack_nint(encoded.payload),1,store_->out_per_expert(),false);
        const auto stride=q_strides_.at(index);
        if (pool.nint.q_expert_stride>stride) throw std::runtime_error("NINT expert exceeded metadata stride");
        if (pool.nint.q_expert_stride!=stride) {
            auto padded=tb::zeros({1,stride},tb::TensorOptions().dtype(tb::kUInt8));
            std::memcpy(padded.data_ptr(),pool.nint.q_packed.data_ptr(),pool.nint.q_packed.numel());
            pool.nint.q_packed=std::move(padded);
            pool.nint.q_expert_stride=stride;
        }
    } else {
        pool.family=MixedMoeFamily::Nvq;
        pool.nvq=to_cpu_nvq(unpack_nvq(encoded.payload,encoded.dtype));
        if (static_cast<std::size_t>(index)<metadata_->pools.size())
            pool.nvq.codebook=metadata_->pools[index].nvq.codebook;
    }
    ++materializations_;
    return pool;
}

void MoeQuantRangeSource::bind_host_cache(std::shared_ptr<MoeHostExpertCache> cache,int source_id) {
    if (!cache || source_id<0 || host_cache_) throw std::logic_error("invalid or repeated expert host cache binding");
    host_cache_=std::move(cache);
    source_id_=source_id;
}

mfq::MoeCacheKey MoeQuantRangeSource::host_key(int expert) const {
    return {source_id_,store_->expert_pool(expert),expert};
}

std::size_t MoeQuantRangeSource::expert_field_bytes(int expert) const {
    return field_bytes_.at(store_->expert_pool(expert));
}

bool MoeQuantRangeSource::gpu_resident(int expert) const {
    (void)store_->expert_pool(expert);
    return gpu_resident_[expert].load();
}

MoeHostExpertCache::Lease MoeQuantRangeSource::acquire_expert(int expert) const {
    const auto bytes=expert_field_bytes(expert);
    if (host_cache_) return host_cache_->acquire(host_key(expert),bytes,[this,expert] {
        auto pool=read_expert(expert);
        if (expert_fields_bytes(pool)!=expert_field_bytes(expert))
            throw std::runtime_error("decoded expert exceeded registered RAM layout");
        return pool;
    },!gpu_resident(expert));
    auto value=std::make_shared<MoeHostExpert>();
    value->weights=read_expert(expert);
    value->bytes=bytes;
    return value;
}

bool MoeQuantRangeSource::demote_expert(int expert,const MoeHostExpertCache::Load& copy) {
    mark_gpu_resident(expert,false);
    return host_cache_ && host_cache_->demote(host_key(expert),expert_field_bytes(expert),[&] {
        auto pool=copy();
        if (expert_fields_bytes(pool)!=expert_field_bytes(expert))
            throw std::runtime_error("GPU eviction exceeded registered RAM layout");
        return pool;
    });
}

void MoeQuantRangeSource::mark_gpu_resident(int expert,bool resident) {
    (void)store_->expert_pool(expert);
    gpu_resident_[expert].store(resident);
    if (resident && host_cache_) host_cache_->promote(host_key(expert));
}

tb::Tensor MoeQuantRangeSource::forward_cpu(CudaExecutionContext& execution,tb::Tensor input,
        const MoeRoutePlan& route) const {
    const int tokens=static_cast<int>(route.ids.size(0)),routes=static_cast<int>(route.ids.size(1));
    if (route.n_experts!=store_->num_experts() || input.size(0)!=tokens || input.size(-1)!=store_->neuron_len() ||
        (input.dim()!=2 && input.dim()!=3) || (input.dim()==3 && input.size(1)!=routes))
        throw std::runtime_error("CPU cold expert input/route shape mismatch");
    const auto device=input.device();
    auto ids=route.ids.to(tb::kCPU).to(tb::kInt32).contiguous();
    auto x=input.to(tb::kCPU).to(tb::kFloat32).contiguous();
    const int width=store_->neuron_len(),output=store_->out_per_expert();
    auto result=tb::empty({tokens,routes,output},tb::TensorOptions().dtype(tb::kFloat16));
    std::vector<std::vector<int>> selected(store_->num_experts());
    for (int position=0; position<tokens*routes; ++position) {
        const auto expert=ids.data_ptr<std::int32_t>()[position];
        if (expert<0 || expert>=store_->num_experts()) throw std::out_of_range("CPU cold expert route ID");
        selected[expert].push_back(position);
    }
    for (int expert=0; expert<store_->num_experts(); ++expert) {
        const auto& positions=selected[expert];
        if (positions.empty()) continue;
        const auto lease=acquire_expert(expert);
        const auto& pool=lease->weights;
        auto batch=tb::empty({static_cast<std::int64_t>(positions.size()),width},tb::TensorOptions().dtype(tb::kFloat32));
        for (std::size_t row=0; row<positions.size(); ++row) {
            const int source=input.dim()==2 ? positions[row]/routes : positions[row];
            std::memcpy(batch.data_ptr<float>()+row*width,x.data_ptr<float>()+std::size_t(source)*width,width*sizeof(float));
        }
        QuantLinear linear;
        linear.logical_out=output; linear.logical_neuron_len=width;
        if (pool.family==MixedMoeFamily::Nint) { linear.kind=QuantLinearKind::Nint; linear.nint=pool.nint; }
        else { linear.kind=QuantLinearKind::Nvq; linear.nvq=pool.nvq; }
        const auto value=linear.forward(execution,batch);
        for (std::size_t row=0; row<positions.size(); ++row)
            std::memcpy(result.data_ptr<mfq_half>()+std::size_t(positions[row])*output,
                value.data_ptr<mfq_half>()+row*output,output*sizeof(mfq_half));
        ++cpu_projections_;
    }
    return result.to(device);
}
