#include "offload.h"
#include "storage/weights/moe/kv_expert_loan.h"
#include "kernels/moe/pipeline.h"

#include <algorithm>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <limits>
#include <tuple>
#include "native/context.h"
#include "native/resident_allocations.h"

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
namespace {
constexpr auto no_offset = std::numeric_limits<std::size_t>::max();
struct Slots {
    std::map<std::size_t,std::size_t> free;
    explicit Slots(std::size_t bytes=0) {if(bytes)free[0]=bytes;}
    std::size_t take(std::size_t bytes) {
        for(auto it=free.begin();it!=free.end();++it)if(it->second>=bytes) {
            const auto offset=it->first,remaining=it->second-bytes;
            free.erase(it);if(remaining)free[offset+bytes]=remaining;return offset;
        }
        return no_offset;
    }
    void give(std::size_t offset,std::size_t bytes) {
        if(offset==no_offset)return;
        auto next=free.lower_bound(offset);
        if(next!=free.begin()) {auto prior=std::prev(next);if(prior->first+prior->second==offset){offset=prior->first;bytes+=prior->second;free.erase(prior);}}
        next=free.lower_bound(offset);
        if(next!=free.end() && offset+bytes==next->first){bytes+=next->second;free.erase(next);}
        free[offset]=bytes;
    }
};
tb::Tensor indices(const std::vector<int64_t>& values, const tb::TensorOptions& options) {
    auto cpu = tb::empty({static_cast<int64_t>(values.size())}, options.device(tb::kCPU).dtype(tb::kInt64));
    if (!values.empty()) std::memcpy(cpu.data_ptr(), values.data(), values.size() * sizeof(int64_t));
    return cpu.to(options.dtype(tb::kInt64));
}
}

struct KvOffloadStore::Block {
    std::uint64_t id = 0;
    std::size_t bytes = 0;
    tb::Tensor gpu;
    tb::TensorOptions options;
    int64_t rows = 0, width = 0;
    cache::QsaKvStore::BlockPtr cold;
    std::size_t gpu_offset = no_offset, ram_offset = no_offset;
};

struct KvOffloadStore::Impl {
    std::function<void(std::size_t)> reserve_ram;
    explicit Impl(KvOffloadConfig value) : config(std::move(value)) {
        if (config.buffer_bytes < 4096 || config.gpu_budget_bytes < config.buffer_bytes)
            throw std::invalid_argument("CUDA KV budget cannot hold its I/O staging buffer");
        if(config.preallocate) {
            ResidentAllocationScope resident;
            gpu_arena=tb::empty({int64_t(hot_limit())},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
            gpu_slots=Slots(hot_limit());
            if(config.ram_preallocated && config.ram_budget_bytes) {
                ram_arena.resize(config.ram_budget_bytes);
                charge_tensor_host_bytes(config.ram_budget_bytes);
            }
            ram_slots=Slots(config.ram_preallocated?config.ram_budget_bytes:0);
            if(!config.ram_preallocated)initialize_disk();
        } else initialize_disk();
    }
    ~Impl() {if(config.preallocate)tensor_host_bytes.fetch_sub(ram_arena.size());}
    void initialize_disk() {
        if(ram)return;
        if(!config.ssd_enabled)throw std::runtime_error("kv_memory_insufficient: available VRAM and RAM cannot hold the configured KV cache; choose automatic with SSD KV or reduce context length");
        cache::QsaKvStoreConfig cold;
        cold.buffer_bytes = config.buffer_bytes;
        cold.budget_bytes = (config.preallocate && config.ram_preallocated?0:config.ram_budget_bytes) + config.buffer_bytes;
        cold.directory = config.directory;
        ram = std::make_shared<cache::QsaKvStore>(std::move(cold));
    }
    std::size_t hot_limit() const { return config.gpu_budget_bytes - config.buffer_bytes; }
    void spill_ram() {
        while(!ram_lru.empty()) {
            auto weak=ram_lru.back();auto block=weak.lock();
            if(!block || block->ram_offset==no_offset){ram_lru.pop_back();continue;}
            initialize_disk();
            if(!block->cold) {
                std::vector<std::uint8_t> payload(block->bytes);
                std::memcpy(payload.data(),ram_arena.data()+block->ram_offset,block->bytes);
                block->cold=ram->write(std::move(payload));
            }
            ram_lru.pop_back();ram_slots.give(block->ram_offset,block->bytes);block->ram_offset=no_offset;
            ram_bytes-=block->bytes;return;
        }
        throw std::runtime_error("kv_memory_insufficient: no free RAM KV blocks");
    }
    void store_cold(const BlockPtr& block) {
        if(block->ram_offset!=no_offset || block->cold)return;
        if(config.preallocate && config.ram_preallocated && block->bytes<=config.ram_budget_bytes) {
            auto offset=ram_slots.take(block->bytes);
            while(offset==no_offset){spill_ram();offset=ram_slots.take(block->bytes);}
            MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(ram_arena.data()+offset,block->gpu.data_ptr(),block->bytes,cudaMemcpyDeviceToHost,current_stream().stream()));
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
            block->ram_offset=offset;ram_bytes+=block->bytes;ram_lru.push_front(block);
        } else {
            initialize_disk();
            if(reserve_ram)reserve_ram(block->bytes*2);
            auto cpu=block->gpu.to(tb::kCPU).contiguous();
            std::vector<std::uint8_t> payload(block->bytes);
            std::memcpy(payload.data(),cpu.data_ptr(),payload.size());
            block->cold=ram->write(std::move(payload));
        }
        d2h_bytes+=block->bytes;
    }
    bool evict_one() {
        while(!lru.empty()) {
            auto found=hot.find(lru.back());auto block=found->second.block.lock();
            if(block) {
                store_cold(block);
                gpu_slots.give(block->gpu_offset,block->bytes);block->gpu_offset=no_offset;
                block->gpu={};gpu_bytes-=block->bytes;++evictions;
            }
            lru.pop_back();hot.erase(found);if(block)return true;
        }
        return false;
    }
    tb::Tensor allocate_gpu(const BlockPtr& block) {
        auto offset=gpu_slots.take(block->bytes);
        while(offset==no_offset && evict_one())offset=gpu_slots.take(block->bytes);
        if(offset==no_offset)throw std::runtime_error("CUDA KV arena cannot hold one block");
        if(expert_loan)expert_loan->protect(loan_region,int64_t(offset+block->bytes));
        block->gpu_offset=offset;
        auto* data=static_cast<std::uint8_t*>(gpu_arena.data_ptr())+offset;
        auto storage=std::make_shared<TensorStorage>();
        storage->base=data;storage->bytes=block->bytes;storage->device=block->options.target_device();
        storage->owner=std::make_shared<tb::Tensor>(gpu_arena);
        const std::vector<int64_t> shape{block->rows,block->width};
        return Tensor(storage,make_contiguous_view(data,shape,block->options.scalar_type(),block->options.target_device()));
    }
    void touch(const BlockPtr& block) {
        auto found = hot.find(block->id);
        if (found != hot.end()) lru.splice(lru.begin(), lru, found->second.order);
        else {
            lru.push_front(block->id);
            hot.emplace(block->id, Hot{block, lru.begin()});
            gpu_bytes += block->bytes;
        }
    }
    void evict() {
        if(config.preallocate){while(gpu_bytes>hot_limit())if(!evict_one())break;return;}
        while (gpu_bytes > hot_limit() && !lru.empty()) {
            const auto id = lru.back();
            auto found = hot.find(id);
            auto block = found->second.block.lock();
            if (block) {
                if (!block->cold) {
                    if (reserve_ram) reserve_ram(block->bytes * 2);
                    auto cpu = block->gpu.to(tb::kCPU).contiguous();
                    std::vector<std::uint8_t> payload(block->bytes);
                    std::memcpy(payload.data(), cpu.data_ptr(), payload.size());
                    block->cold = ram->write(std::move(payload));
                    d2h_bytes += block->bytes;
                }
                block->gpu = {};
                gpu_bytes -= block->bytes;
                ++evictions;
            }
            lru.pop_back(); hot.erase(found);
        }
    }
    void release(Block* block) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        auto found = hot.find(block->id);
        if (found != hot.end()) {
            gpu_bytes -= block->bytes;
            lru.erase(found->second.order); hot.erase(found);
        }
        if(config.preallocate) {
            gpu_slots.give(block->gpu_offset,block->bytes);
            if(block->ram_offset!=no_offset){ram_slots.give(block->ram_offset,block->bytes);ram_bytes-=block->bytes;}
        }
    }
    struct Hot { std::weak_ptr<Block> block; std::list<std::uint64_t>::iterator order; };
    KvOffloadConfig config;
    std::shared_ptr<cache::QsaKvStore> ram;
    tb::Tensor gpu_arena;
    std::shared_ptr<KvExpertLoan> expert_loan;
    std::size_t loan_region=0;
    std::vector<std::uint8_t> ram_arena;
    Slots gpu_slots,ram_slots;
    std::list<std::weak_ptr<Block>> ram_lru;
    std::size_t ram_bytes=0;
    std::uint64_t ram_hits=0;
    mutable std::recursive_mutex mutex;
    std::list<std::uint64_t> lru;
    std::unordered_map<std::uint64_t, Hot> hot;
    std::map<std::tuple<int64_t,int64_t,int,int64_t,int64_t>,std::shared_ptr<GatherWorkspace>> gather_workspaces;
    std::size_t gpu_bytes = 0;
    std::uint64_t next_id = 0, hits = 0, h2d_bytes = 0, d2h_bytes = 0, evictions = 0;
};

KvOffloadStore::KvOffloadStore(KvOffloadConfig config) : impl_(std::make_shared<Impl>(std::move(config))) {}
KvOffloadStore::~KvOffloadStore() = default;
std::shared_ptr<KvOffloadStore::GatherWorkspace> KvOffloadStore::gather_workspace(int64_t rows,int64_t width,tb::ScalarType dtype,int64_t heads,int64_t key_width) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    if(rows<=0 || width<=0 || heads<=0 || width%heads || key_width<0 || (key_width && key_width>=width/heads))
        throw std::invalid_argument("invalid CUDA KV gather geometry");
    auto& workspace=impl_->gather_workspaces[{rows,width,int(dtype),heads,key_width}];
    if(!workspace) {
        ResidentAllocationScope resident;
        auto fresh=std::make_shared<GatherWorkspace>();
        fresh->heads=heads;fresh->key_width=key_width;
        const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(dtype);
        fresh->rows=tb::zeros({rows,width},options);
        fresh->upload=tb::empty({rows,width},options);
        fresh->host=tb::empty({rows,width},options.device(tb::kCPU).pinned_memory(true));
        const auto fields=key_width?2*heads:1;
        fresh->descriptors=tb::zeros({4+rows*fields*4},options.device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true));
        fresh->device_descriptors=tb::zeros({4+rows*fields*4},options.dtype(tb::kInt64));
        workspace=std::move(fresh);
    }
    return workspace;
}
std::size_t KvOffloadStore::graph_workspace_bytes() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    std::size_t bytes=0;
    for(const auto& [key,work]:impl_->gather_workspaces)bytes+=work->rows.nbytes()+work->upload.nbytes()+work->device_descriptors.nbytes();
    return bytes;
}
const KvOffloadConfig& KvOffloadStore::config() const { return impl_->config; }

KvOffloadStore::BlockPtr KvOffloadStore::write(const tb::Tensor& rows) {
    if (!rows.is_cuda() || rows.dim() != 2 || !rows.numel() || (rows.scalar_type() != tb::kFloat16 && rows.scalar_type() != tb::kInt32))
        throw std::invalid_argument("CUDA KV microblock must contain nonempty FP16 or packed GPU rows");
    const auto bytes = static_cast<std::size_t>(rows.numel()) * rows.element_size();
    if (bytes > impl_->config.buffer_bytes / 2)
        throw std::invalid_argument("CUDA KV microblock exceeds its write staging buffer");
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    std::weak_ptr<Impl> owner = impl_;
    auto block = BlockPtr(new Block, [owner](Block* value) {
        if (auto impl = owner.lock()) impl->release(value);
        delete value;
    });
    block->id = ++impl_->next_id;
    block->bytes = bytes; block->rows = rows.size(0); block->width = rows.size(1);
    block->options = rows.options();
    if(impl_->config.preallocate && bytes>impl_->hot_limit()) {
        block->gpu=rows.contiguous();impl_->store_cold(block);block->gpu={};++impl_->evictions;
        return block;
    }
    if(impl_->config.preallocate) {
        block->gpu=impl_->allocate_gpu(block);block->gpu.copy_(rows);
        // A slot can be reused by another write on this same stream.
    } else block->gpu = rows.contiguous().clone();
    impl_->touch(block); impl_->evict();
    return block;
}

tb::Tensor KvOffloadStore::read(const BlockPtr& block) {
    if (!block) throw std::invalid_argument("missing CUDA KV block");
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    if (block->gpu.defined()) { ++impl_->hits; impl_->touch(block); return block->gpu; }
    if(impl_->config.preallocate) {
        // Keep read results outside the eviction arena: a gather may retain
        // this tensor while subsequent reads recycle other arena slots.
        auto result=tb::empty({block->rows,block->width},block->options);
        if(block->ram_offset!=no_offset) {
            MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(result.data_ptr(),impl_->ram_arena.data()+block->ram_offset,block->bytes,cudaMemcpyHostToDevice,current_stream().stream()));
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
            impl_->ram_lru.remove_if([&](const auto& weak){auto value=weak.lock();return !value || value==block;});
            impl_->ram_lru.push_front(block);++impl_->ram_hits;
        } else {
            if(!impl_->ram || !block->cold)throw std::runtime_error("missing spilled KV block");
            auto payload=impl_->ram->read(block->cold);
            MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(result.data_ptr(),payload->data(),block->bytes,cudaMemcpyHostToDevice,current_stream().stream()));
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
        }
        impl_->h2d_bytes+=block->bytes;return result;
    }
    if (impl_->reserve_ram) impl_->reserve_ram(block->bytes * 2);
    auto payload = impl_->ram->read(block->cold);
    const auto options = block->options;
    auto cpu = tb::empty({block->rows, block->width}, options.device(tb::kCPU));
    std::memcpy(cpu.data_ptr(), payload->data(), block->bytes);
    auto gpu = cpu.to(block->options);
    impl_->h2d_bytes += block->bytes;
    if (block->bytes <= impl_->hot_limit()) {
        block->gpu = gpu; impl_->touch(block); impl_->evict();
    }
    return gpu;
}

void KvOffloadStore::flush() { if(impl_->ram)impl_->ram->flush(); }
void KvOffloadStore::set_ram_reservation(std::function<void(std::size_t)> reserve) {
    impl_->reserve_ram = std::move(reserve);
}
void KvOffloadStore::set_expert_loan(std::shared_ptr<KvExpertLoan> loan) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    if(!loan || !impl_->gpu_arena.defined() || !impl_->gpu_arena.nbytes())return;
    impl_->loan_region=loan->add(impl_->gpu_arena,int64_t(impl_->gpu_arena.nbytes()),1,false,1,true);
    impl_->expert_loan=std::move(loan);
}
std::size_t KvOffloadStore::expert_loan_bytes() const {
    return impl_->expert_loan?impl_->expert_loan->region_bytes(impl_->loan_region):0;
}
std::vector<std::pair<std::string, double>> KvOffloadStore::metrics() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    const auto cold = impl_->ram ? impl_->ram->stats() : cache::QsaKvStoreStats{};
    return {{"qsa_kv_offload_enabled", 1.0}, {"qsa_kv_ssd_enabled",impl_->config.ssd_enabled?1.0:0.0},
        {"qsa_kv_graph_workspace_bytes",double(graph_workspace_bytes())},
        {"qsa_kv_resident_bytes", double((impl_->config.preallocate?impl_->hot_limit():impl_->gpu_bytes)-expert_loan_bytes())},
        {"qsa_kv_budget_bytes", double(impl_->config.gpu_budget_bytes)},
        {"qsa_kv_ram_enabled", impl_->config.ram_budget_bytes ? 1.0 : 0.0},
        {"qsa_kv_ram_bytes", double(impl_->config.preallocate && impl_->config.ram_preallocated?impl_->ram_arena.size():cold.hot_bytes)}, {"qsa_kv_ram_limit_bytes", double(impl_->config.ram_budget_bytes)},
        {"qsa_kv_ssd_bytes", double(cold.disk_bytes)}, {"qsa_kv_pending_bytes", double(cold.pending_bytes)},
        {"qsa_kv_ssd_reads", double(cold.reads)}, {"qsa_kv_ram_hits", double(cold.hits+impl_->ram_hits)},
        {"qsa_kv_gpu_hits", double(impl_->hits)}, {"qsa_kv_ssd_read_bytes", double(cold.read_bytes)},
        {"qsa_kv_ssd_written_bytes", double(cold.written_bytes)},
        {"qsa_kv_h2d_bytes", double(impl_->h2d_bytes)}, {"qsa_kv_d2h_bytes", double(impl_->d2h_bytes)},
        {"qsa_kv_gpu_evictions", double(impl_->evictions)}};
}

KvOffloadSequence::KvOffloadSequence(std::shared_ptr<KvOffloadStore> store, int64_t width, int64_t block_rows, tb::ScalarType dtype)
    : store_(std::move(store)), width_(width), block_rows_(block_rows), dtype_(dtype) {
    if (!store_ || width <= 0 || block_rows <= 0 || (dtype != tb::kFloat16 && dtype != tb::kInt32) ||
        std::size_t(width) * block_rows * (dtype == tb::kInt32 ? 4 : 2) > store_->config().buffer_bytes / 2)
        throw std::invalid_argument("invalid CUDA KV sequence geometry");
}
void KvOffloadSequence::reset() { blocks_.clear(); position_ = 0; }
void KvOffloadSequence::append(const tb::Tensor& source) {
    if (!source.is_cuda() || source.dim() != 2 || source.size(1) != width_ || source.size(0) <= 0)
        throw std::invalid_argument("invalid CUDA KV rows");
    const auto offset = position_;
    const auto previous_count = blocks_.size();
    const auto previous_tail = position_ % block_rows_ ? blocks_.back() : KvOffloadStore::BlockPtr{};
    try {
        auto rows = source.to(dtype_);
        if (position_ % block_rows_) {
            rows = tb::cat({store_->read(blocks_.back()), rows}, 0);
            blocks_.pop_back();
        }
        for (int64_t begin = 0; begin < rows.size(0); begin += block_rows_)
            blocks_.push_back(store_->write(rows.narrow(0, begin, std::min(block_rows_, rows.size(0) - begin))));
        position_ += source.size(0);
    } catch (...) {
        blocks_.resize(previous_count - (previous_tail ? 1 : 0));
        if (previous_tail) blocks_.push_back(previous_tail);
        position_ = offset; throw;
    }
}
void KvOffloadSequence::truncate(int64_t position) {
    if (position < 0 || position > position_) throw std::out_of_range("CUDA KV trim exceeds its sequence");
    if (position == position_) return;
    const auto full = position / block_rows_, tail = position % block_rows_;
    KvOffloadStore::BlockPtr remainder;
    if (tail) remainder = store_->write(store_->read(blocks_.at(full)).narrow(0, 0, tail));
    blocks_.resize(full);
    if (remainder) blocks_.push_back(std::move(remainder));
    position_ = position;
}
tb::Tensor KvOffloadSequence::gather(const std::vector<int64_t>& rows) const {
    if (rows.empty() || blocks_.empty()) throw std::invalid_argument("empty CUDA KV selection");
    std::map<int64_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> groups;
    for (std::size_t output = 0; output < rows.size(); ++output) {
        if (rows[output] < 0 || rows[output] >= position_) throw std::out_of_range("CUDA KV selected row exceeds its sequence");
        auto& group = groups[rows[output] / block_rows_];
        group.first.push_back(rows[output] % block_rows_); group.second.push_back(output);
    }
    const auto options = blocks_.front()->options;
    auto result = tb::empty({static_cast<int64_t>(rows.size()), width_}, options);
    for (const auto& [block, group] : groups) {
        auto payload = store_->read(blocks_.at(block));
        auto selected = payload.index_select(0, indices(group.first, options));
        result.index_copy_(0, indices(group.second, options), selected);
    }
    return result;
}
void KvOffloadSequence::gather_into(const std::vector<int64_t>& rows,const KvOffloadStore::GatherWorkspace& work) const {
    if(rows.empty() || int64_t(rows.size())>work.rows.size(0) || work.rows.size(1)!=width_ || work.rows.scalar_type()!=dtype_)
        throw std::invalid_argument("invalid fixed CUDA KV gather workspace");
    auto& impl=*store_->impl_;
    std::lock_guard<std::recursive_mutex> lock(impl.mutex);
    const auto row_bytes=std::size_t(width_)*work.rows.element_size();
    auto* host=static_cast<uint8_t*>(work.host.data_ptr());
    auto* upload=static_cast<uint8_t*>(work.upload.data_ptr());
    auto* destination=static_cast<uint8_t*>(work.rows.data_ptr());
    auto* header=work.descriptors.data_ptr<int64_t>();
    auto* copies=reinterpret_cast<MoeMappedCopyDescriptor*>(header+4);
    std::map<int64_t,std::vector<std::pair<int64_t,std::size_t>>> groups;
    for(std::size_t i=0;i<rows.size();++i) {
        if(rows[i]<0 || rows[i]>=position_)throw std::out_of_range("CUDA KV selected row exceeds its sequence");
        groups[rows[i]/block_rows_].emplace_back(rows[i]%block_rows_,i);
    }
    std::size_t cold_bytes=0,count=0,total=0;
    for(const auto& [index,selected]:groups) {
        const auto& block=blocks_.at(index);
        const uint8_t* source=nullptr;
        std::shared_ptr<const std::vector<uint8_t>> payload;
        const bool hot=block->gpu.defined();
        if(hot) {source=static_cast<const uint8_t*>(block->gpu.data_ptr());impl.touch(block);++impl.hits;}
        else if(block->ram_offset!=no_offset) {
            source=impl.ram_arena.data()+block->ram_offset;++impl.ram_hits;
            impl.ram_lru.remove_if([&](const auto& weak){auto value=weak.lock();return !value || value==block;});
            impl.ram_lru.push_front(block);
        } else {
            if(!impl.ram || !block->cold)throw std::runtime_error("missing spilled KV block");
            payload=impl.ram->read(block->cold);source=payload->data();
        }
        for(const auto& [row,output]:selected) {
            const auto* input=source+std::size_t(row)*row_bytes;
            if(!hot) {std::memcpy(host+cold_bytes,input,row_bytes);input=upload+cold_bytes;cold_bytes+=row_bytes;}
            const auto add=[&](uint8_t* target,const uint8_t* source,std::size_t bytes) {
                copies[count++]={reinterpret_cast<uint64_t>(target),reinterpret_cast<uint64_t>(source),total,bytes};
                total+=(bytes+15)&~std::size_t(15);
            };
            if(!work.key_width)add(destination+output*row_bytes,input,row_bytes);
            else {
                const auto keys=std::size_t(work.key_width)*work.rows.element_size();
                const auto values=row_bytes/work.heads-keys,capacity=std::size_t(work.rows.size(0));
                for(int64_t head=0;head<work.heads;++head) {
                    add(destination+(head*capacity+output)*keys,input+head*keys,keys);
                    add(destination+capacity*work.heads*keys+(head*capacity+output)*values,
                        input+work.heads*keys+head*values,values);
                }
            }
        }
    }
    header[0]=int64_t(count);header[1]=int64_t(total);header[2]=header[3]=0;
    const auto stream=current_stream().stream();
    if(cold_bytes)MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(upload,host,cold_bytes,cudaMemcpyHostToDevice,stream));
    impl.h2d_bytes+=cold_bytes;
    MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(work.device_descriptors.data_ptr(),header,(4+count*4)*sizeof(int64_t),cudaMemcpyHostToDevice,stream));
    moe_copy_mapped(work.device_descriptors.data_ptr(),reinterpret_cast<const uint32_t*>(work.device_descriptors.data_ptr<int64_t>()+2),
        int(std::min<std::size_t>(256,(total+4095)/4096)),256,stream);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
}
tb::Tensor KvOffloadSequence::range(int64_t start, int64_t count) const {
    if (start < 0 || count <= 0 || start > position_ || count > position_ - start)
        throw std::out_of_range("CUDA KV range exceeds its sequence");
    auto result = tb::empty({count, width_}, blocks_.front()->options);
    for (int64_t written = 0; written < count;) {
        const auto row = start + written, block = row / block_rows_, offset = row % block_rows_;
        auto source = store_->read(blocks_.at(block));
        const auto n = std::min(count-written,source.size(0)-offset);
        result.narrow(0,written,n).copy_(source.narrow(0,offset,n));
        written += n;
    }
    return result;
}
}
