#include "kv_memory.h"
#include "kv_vmm.h"
#include "kv_budget.h"
#include "kv_expert_loan.h"
#include "execution_options.h"
#include "mfq_cuda_context.h"
#include "mfq_cuda_resident_allocations.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <filesystem>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
namespace {
constexpr const char* insufficient = "kv_memory_insufficient: available VRAM and RAM cannot hold the configured KV cache; choose automatic with SSD KV or reduce context length";
std::size_t setting(const char* name, std::size_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const auto result = std::strtoull(value, &end, 10);
    if (*value == '-' || !end || *end) throw std::invalid_argument(std::string("invalid ") + name);
    return std::size_t(result);
}
std::size_t host_available() {
#ifdef _WIN32
    MEMORYSTATUSEX status{}; status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? std::size_t(status.ullAvailPhys) : 0;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES), size = sysconf(_SC_PAGESIZE);
    return pages > 0 && size > 0 ? std::size_t(pages) * size : 0;
#endif
}
std::size_t width_bytes(const tb::TensorOptions& options) { return options.scalar_type() == tb::kInt32 ? 4 : 2; }
bool discrete_device() {
    if(const auto* value=std::getenv("MFQ_MEMORY_ARCHITECTURE"))return std::string(value)=="discrete";
    int device=0,integrated=0,host_tables=0;
    MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&integrated,cudaDevAttrIntegrated,device));
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&host_tables,cudaDevAttrPageableMemoryAccessUsesHostPageTables,device));
    return !integrated && !host_tables;
}
tb::Tensor external(void* data, std::shared_ptr<void> owner, const std::vector<int64_t>& shape,
                    const tb::TensorOptions& options) {
    auto view = make_contiguous_view(data, shape, options.scalar_type(), options.target_device());
    auto storage = std::make_shared<TensorStorage>();
    storage->base = data; storage->bytes = view.nbytes(); storage->device = options.target_device();
    storage->owner = std::move(owner);
    return Tensor(std::move(storage), view);
}
tb::Tensor host_tensor(const std::vector<int64_t>& shape, const tb::TensorOptions& options) {
    std::size_t bytes = width_bytes(options);
    for (const auto dimension : shape) {
        if (dimension <= 0 || std::size_t(dimension) > std::numeric_limits<std::size_t>::max()/bytes)
            throw std::invalid_argument("KV tensor size overflow");
        bytes *= dimension;
    }
    void* host = nullptr;
    MFQ_NATIVE_CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
    charge_tensor_host_bytes(bytes);
    auto owner = std::shared_ptr<void>(host, [bytes](void* p) {
        (void)cudaDeviceSynchronize(); (void)cudaFreeHost(p); tensor_host_bytes.fetch_sub(bytes);
    });
    std::memset(host, 0, bytes);
    void* gpu = nullptr; MFQ_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&gpu, host, 0));
    return external(gpu, std::move(owner), shape, options);
}
}

struct KvMemoryAllocation::Virtual {
    kv_vmm::VmmRange range;
    int64_t maximum, heads, width, committed = 0, mapping_chunk;
    std::size_t element;
    Virtual(int64_t maximum, int64_t heads, int64_t width, std::size_t element, int64_t mapping_chunk)
        : maximum(maximum), heads(heads), width(width), mapping_chunk(mapping_chunk), element(element) {
        if (!range.reserve(std::size_t(maximum)*heads*width*element))
            throw std::runtime_error("kv_dynamic_unavailable: CUDA virtual memory reservation failed");
    }
    ~Virtual() { (void)cudaDeviceSynchronize(); }
    std::set<int64_t> missing(int64_t tokens) const {
        std::set<int64_t> result;
        tokens=std::min(maximum,((tokens+mapping_chunk-1)/mapping_chunk)*mapping_chunk);
        if (tokens <= committed) return result;
        const auto g = kv_vmm::vmm_granularity();
        for (int64_t head=0; head<heads; ++head) {
            const auto begin = std::size_t(head)*maximum*width*element;
            const auto end = begin + std::size_t(tokens)*width*element;
            for (auto page=begin/g; page<(end+g-1)/g; ++page)
                if (!range.mapped(int64_t(page))) result.insert(int64_t(page));
        }
        return result;
    }
    void ensure(int64_t tokens) {
        tokens=std::min(maximum,((tokens+mapping_chunk-1)/mapping_chunk)*mapping_chunk);
        const auto fresh = missing(tokens);
        if (!fresh.empty()) MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
        for (const auto page : fresh) {
            if (!range.map_range(page,page+1,[]{return kv_vmm::VmmChunk(0);}))
                throw std::runtime_error(insufficient);
            MFQ_NATIVE_CUDA_CHECK(cudaMemsetAsync(range.base()+page*kv_vmm::vmm_granularity(),0,
                kv_vmm::vmm_granularity(),current_stream().stream()));
        }
        committed = std::max(committed,tokens);
    }
};

KvMemoryOptions kv_memory_options() {
    KvMemoryOptions o;
    if (const auto* p=std::getenv("MFQ_KV_PLACEMENT")) o.placement=p;
    if (o.placement!="auto" && o.placement!="auto_ssd" && o.placement!="vram" && o.placement!="ram")
        throw std::invalid_argument("invalid KV placement");
    if (const auto* p=std::getenv("MFQ_KV_ALLOCATION")) {
        const std::string allocation=p;
        if (allocation!="preallocated" && allocation!="dynamic") throw std::invalid_argument("invalid KV allocation");
        o.dynamic=allocation=="dynamic";
    }
    if (o.dynamic && (o.placement=="ram" || o.placement=="auto_ssd"))
        throw std::invalid_argument("kv_dynamic_unavailable: experimental dynamic KV requires VRAM or automatic without SSD");
    o.gpu_limit=setting("MFQ_KV_GPU_LIMIT_BYTES",0);
    o.gpu_budget=setting("MFQ_KV_GPU_BUDGET_BYTES",0);
    o.budget_policy=std::getenv("MFQ_KV_RAM_PREALLOCATED") || o.gpu_budget;
    o.ram_preallocated=setting("MFQ_KV_RAM_PREALLOCATED",1)!=0;
    if(!o.ram_preallocated && o.placement!="auto_ssd")throw std::invalid_argument("kv_ssd_required: disabling RAM preallocation requires SSD KV");
    o.ram_limit_set=std::getenv("MFQ_KV_RAM_LIMIT_BYTES")!=nullptr;
    o.ram_limit=setting("MFQ_KV_RAM_LIMIT_BYTES",0);
    o.auto_stream_threshold=int64_t(setting("MFQ_KV_AUTO_STREAM_MIN_CONTEXT",65536));
    o.resident_cells=int64_t(setting("MFQ_KV_RESIDENT_CELLS",32768));
    if (const auto* p=std::getenv("MFQ_RUNTIME_PREFIX_CACHE_DIR")) o.directory=p;
    return o;
}
std::size_t KvMemoryAllocation::raw_bytes() const {return std::size_t(maximum)*heads*(key_width+value_width)*width_bytes(options);}
std::size_t KvMemoryAllocation::index_bytes() const {return std::size_t(maximum+(maximum+pool-1)/pool)*index_width*2;}
std::size_t KvMemoryAllocation::gpu_bytes() const {
    if (!ready) return 0;
    if (virtual_keys) return index_bytes() + std::size_t(virtual_keys->range.mapped_count()+virtual_values->range.mapped_count())*kv_vmm::vmm_granularity();
    return index_bytes() + hot.bytes() + (host || offload ? 0 : raw_bytes());
}
std::size_t KvMemoryAllocation::ram_bytes() const {
    if(!ready)return 0;
    if(hot.exclusive())return std::size_t(hot.cold_keys.numel()+hot.cold_values.numel())*width_bytes(options);
    return host?raw_bytes():0;
}
std::size_t KvMemoryAllocation::growth_bytes(int64_t tokens) const {
    tokens=std::min(maximum,tokens);
    return virtual_keys ? (virtual_keys->missing(tokens).size()+virtual_values->missing(tokens).size())*kv_vmm::vmm_granularity() : 0;
}
void KvMemoryAllocation::ensure(int64_t tokens) {
    if (tokens<0 || tokens>maximum) throw std::invalid_argument("KV growth exceeds configured context");
    if(auto loan=expert_loan.lock())loan->ensure(tokens+pool-1);
    if (virtual_keys) {virtual_keys->ensure(tokens); virtual_values->ensure(tokens);}
}
KvMemoryManager::KvMemoryManager(KvMemoryOptions options):options_(std::move(options)) {}
std::shared_ptr<KvMemoryAllocation> KvMemoryManager::add(int64_t maximum,int64_t heads,int64_t kw,int64_t vw,
        int64_t iw,int64_t pool,const tb::TensorOptions& options,int64_t selected_columns) {
    if (ready_) {
        auto policy=options_;policy.additional_request=true;
        if(policy.budget_policy && policy.gpu_budget)for(const auto& weak:allocations_)if(auto existing=weak.lock()) {
            if(existing->maximum!=maximum || existing->heads!=heads || existing->key_width!=kw || existing->value_width!=vw)continue;
            policy.gpu_budget=existing->index_bytes()+(existing->hot.exclusive()?
                std::size_t(existing->hot.keys.numel()+existing->hot.values.numel())*width_bytes(existing->options):
                existing->host || existing->offload?0:existing->raw_bytes());
            break;
        }
        // Independent graph sessions require directly addressable raw caches.
        // Admit extra GPU sessions against currently free memory.
        if(!automatic_ram_ && !policy.budget_policy && (policy.placement=="auto" || policy.placement=="auto_ssd"))policy.placement="vram";
        KvMemoryManager child(policy);
        auto a=child.add(maximum,heads,kw,vw,iw,pool,options,selected_columns);
        child.finalize();allocations_.push_back(a);return a;
    }
    auto a=std::make_shared<KvMemoryAllocation>();
    a->maximum=maximum;a->heads=heads;a->key_width=kw;a->value_width=vw;a->index_width=iw;a->pool=pool;a->options=options;
    a->selected_columns=selected_columns;
    allocations_.push_back(a);return a;
}
void KvMemoryManager::finalize(std::size_t minimum_experts) {
    if (ready_) return;
    if(!options_.ram_preallocated && options_.placement!="auto_ssd")throw std::invalid_argument("kv_ssd_required: disabling RAM preallocation requires SSD KV");
    std::size_t raw=0,index=0;
    for (auto& weak:allocations_) if(auto a=weak.lock()){raw+=a->raw_bytes();index+=a->index_bytes();}
    if (!raw) {ready_=true;return;}
    auto context=default_context();context->reclaim_idle_memory();
    const auto free=context->available_memory();
    const auto reserve=(options_.additional_request?std::size_t(256)<<20:runtime_options::workspace_reserve_bytes())+minimum_experts;
    if (free<reserve || index>free-reserve) throw std::runtime_error(insufficient);
    auto gpu=free-reserve-index;
    if(options_.gpu_limit)gpu=std::min(gpu,options_.gpu_limit);
    auto available=host_available();
    std::size_t ram=available> (1ULL<<30) ? available-(1ULL<<30) : 0;
    if(options_.ram_limit_set)ram=std::min(ram,options_.ram_limit);
    // Long discrete-memory sessions keep canonical raw KV in RAM. The GPU
    // cache is bounded by resident cells; selected rows are resolved on-device
    // and the same fixed addresses remain valid in decode/MTP CUDA Graphs.
    int64_t maximum=0,cells=options_.resident_cells;
    std::size_t per_cell=0,fixed_metadata=0;
    for(auto& weak:allocations_)if(auto a=weak.lock()) {
        maximum=std::max(maximum,a->maximum);cells=std::min(cells,a->maximum);
        per_cell+=std::size_t(a->heads)*(a->key_width+a->value_width)*width_bytes(a->options);
        fixed_metadata+=std::size_t(a->maximum)*16+56;
    }
    const bool automatic=options_.placement=="auto" || options_.placement=="auto_ssd";
    const bool tiered_budget=options_.budget_policy && discrete_device();
    // A small target context must not shrink staging below the selected KV
    // workspace. That would turn an 8K prefill into thousands of per-row
    // gathers even though the fixed decode workspace can hold a whole layer.
    std::size_t stream_buffer=std::min<std::size_t>(64ULL<<20,std::max<std::size_t>(4096,raw/16));
    if(tiered_budget)for(const auto& weak:allocations_)if(auto a=weak.lock()) {
        stream_buffer=std::max(stream_buffer,std::size_t(16*a->selected_columns*a->heads)*
            (a->key_width+a->value_width)*width_bytes(a->options));
    }
    bool exclusive=false,raw_host_only=false;
    if(tiered_budget) {
        if(options_.gpu_budget && options_.gpu_budget<index)
            throw std::runtime_error("kv_budget_below_indexer: VRAM KV budget must cover the complete target indexer cache");
        const auto desired=options_.gpu_budget?options_.gpu_budget-index:
            (options_.placement=="vram" || options_.dynamic?raw:options_.placement=="ram"?0:std::size_t(std::ceil(raw*kv_resident_fraction(maximum))));
        if(options_.gpu_budget && desired>gpu)throw std::runtime_error("kv_vram_insufficient: explicit KV VRAM budget exceeds available capacity after weights and workspace");
        gpu=std::min(gpu,desired);
        if(options_.placement=="ram")gpu=0;
        if(!options_.dynamic && options_.placement!="vram" && gpu<raw) {
            cells=std::min<int64_t>(maximum,gpu/per_cell);
            gpu=std::size_t(cells)*per_cell;
            const auto required=raw-gpu;
            if(options_.placement!="auto_ssd" && required>ram)throw std::runtime_error(insufficient);
            if(options_.ram_preallocated && required<=ram) {
                exclusive=cells>0;raw_host_only=cells==0;
            } else {
                if(options_.placement!="auto_ssd")throw std::runtime_error(insufficient);
                const auto disk_needed=options_.ram_preallocated?required-std::min(required,ram):required;
                auto directory=std::filesystem::path(options_.directory);
                if(directory.empty())throw std::runtime_error("kv_ssd_insufficient: SSD KV directory is required");
                while(!std::filesystem::exists(directory) && directory.has_parent_path())directory=directory.parent_path();
                if(std::filesystem::space(directory).available<disk_needed+(64ULL<<20))
                    throw std::runtime_error("kv_ssd_insufficient: insufficient disk space for the target cold KV capacity");
                KvOffloadConfig config;
                config.buffer_bytes=stream_buffer;
                config.gpu_budget_bytes=gpu+config.buffer_bytes;
                config.ram_budget_bytes=std::min(ram,required+config.buffer_bytes);
                config.preallocate=true;config.ram_preallocated=options_.ram_preallocated;config.ssd_enabled=true;config.directory=options_.directory;
                offload_=std::make_shared<KvOffloadStore>(config);
                std::cerr<<"runtime_kv_warning code=kv_ssd_overflow target_spill_bytes="<<disk_needed
                    <<" message=KV may spill to SSD; long-context latency and disk usage will increase\n";
            }
        }
    }
    automatic_ram_=!tiered_budget && automatic && !options_.dynamic && options_.auto_stream_threshold>0 &&
        maximum>=options_.auto_stream_threshold && raw<=ram && cells>0 && discrete_device();
    if(automatic_ram_) {
        const auto allocations=std::count_if(allocations_.begin(),allocations_.end(),[](const auto& weak){return !weak.expired();});
        const auto bytes_per_cell=per_cell+std::size_t(allocations)*12;
        if(gpu<=fixed_metadata)automatic_ram_=false;
        else cells=std::min<int64_t>(cells,(gpu-fixed_metadata)/bytes_per_cell);
        if(!cells)automatic_ram_=false;
    }
    if(options_.placement=="vram" && raw>gpu)throw std::runtime_error("kv_vram_insufficient: configured KV cache does not fit in available VRAM");
    if(options_.placement=="ram" && raw>ram)throw std::runtime_error(insufficient);
    if(options_.placement=="auto" && (raw>gpu && raw-gpu>ram))throw std::runtime_error(insufficient);
    if(options_.dynamic && !kv_vmm::vmm_available())throw std::runtime_error("kv_dynamic_unavailable: CUDA virtual memory management is unavailable");
    if(options_.dynamic && raw>gpu)throw std::runtime_error("kv_dynamic_unavailable: experimental dynamic KV currently requires a GPU-resident cache");
    if(!tiered_budget && !automatic_ram_ && !options_.dynamic && options_.placement!="ram" && raw>gpu) {
        KvOffloadConfig config;
        config.buffer_bytes=std::min<std::size_t>(64ULL<<20,std::max<std::size_t>(4096,raw/16));
        // Streaming arenas are preallocated. SSD is opened only on first spill.
        config.gpu_budget_bytes=std::min(raw,gpu);
        if(config.gpu_budget_bytes<config.buffer_bytes*2)throw std::runtime_error(insufficient);
        config.ram_budget_bytes=std::min(ram,raw-(config.gpu_budget_bytes-config.buffer_bytes));
        // A partially filled block is replaced transactionally during append.
        // Retain room for that replacement in addition to the complete cache.
        const auto required_ram=raw-(config.gpu_budget_bytes-config.buffer_bytes)+config.buffer_bytes;
        if(options_.placement=="auto" && required_ram>ram)throw std::runtime_error(insufficient);
        config.ram_budget_bytes=std::min(ram,required_ram);
        config.directory=options_.directory;config.preallocate=true;config.ssd_enabled=options_.placement=="auto_ssd";
        offload_=std::make_shared<KvOffloadStore>(config);
    }
    ResidentAllocationScope resident;
    if(tiered_budget && !options_.additional_request && !options_.dynamic && setting("MFQ_KV_EXPERT_LOAN",1))expert_loan_=std::make_shared<KvExpertLoan>();
    for(auto& weak:allocations_)if(auto a=weak.lock()) {
        const auto shape=[&](int64_t w){return std::vector<int64_t>{1,a->heads,a->maximum,w};};
        auto index_options=a->options;index_options.dtype(tb::kFloat16);
        a->index=tb::zeros({1,a->maximum,a->index_width},index_options);
        a->pooled=tb::zeros({1,(a->maximum+a->pool-1)/a->pool,a->index_width},index_options);
        a->host=options_.placement=="ram" || automatic_ram_ || exclusive || raw_host_only;a->offload=offload_;
        if(options_.dynamic) {
            const auto mapping_chunk = options_.additional_request ? int64_t(1) : int64_t(8192);
            a->virtual_keys = std::make_shared<KvMemoryAllocation::Virtual>(
                a->maximum, a->heads, a->key_width, width_bytes(a->options), mapping_chunk);
            a->virtual_values = std::make_shared<KvMemoryAllocation::Virtual>(
                a->maximum, a->heads, a->value_width, width_bytes(a->options), mapping_chunk);
            a->keys=external(a->virtual_keys->range.base(),a->virtual_keys,shape(a->key_width),a->options);
            a->values=external(a->virtual_values->range.base(),a->virtual_values,shape(a->value_width),a->options);
            // Additional dynamic requests map only admitted prefill/decode
            // pages. A 16K warm allocation would bypass the KV admission gate.
            const auto initial = options_.additional_request ? int64_t(0) : std::min<int64_t>(a->maximum,16384);
            a->ensure(initial);
        } else if(exclusive) {
            auto cold_shape=[&](int64_t width){return std::vector<int64_t>{1,a->heads,a->maximum-cells,width};};
            a->hot=HotKvCache::allocate(a->heads,a->maximum,cells,a->key_width,a->value_width,a->options);
            auto cold_keys=host_tensor(cold_shape(a->key_width),a->options),cold_values=host_tensor(cold_shape(a->value_width),a->options);
            a->hot.set_cold_storage(cold_keys,cold_values);
            // Logical geometry only: tier-aware kernels resolve every row via
            // the maps. The owner retains the smaller physical RAM tensors.
            a->keys=external(cold_keys.data_ptr(),std::make_shared<tb::Tensor>(cold_keys),shape(a->key_width),a->options);
            a->values=external(cold_values.data_ptr(),std::make_shared<tb::Tensor>(cold_values),shape(a->value_width),a->options);
        } else if(a->host) {
            a->keys=host_tensor(shape(a->key_width),a->options);a->values=host_tensor(shape(a->value_width),a->options);
            if(automatic_ram_)a->hot=HotKvCache::allocate(a->heads,a->maximum,cells,a->key_width,a->value_width,a->options);
        } else if(!offload_) {
            a->keys=tb::zeros(shape(a->key_width),a->options);a->values=tb::zeros(shape(a->value_width),a->options);
        }
        a->ready=true;
        if(expert_loan_) {
            a->expert_loan=expert_loan_;
            expert_loan_->add(a->index,a->maximum,std::size_t(a->index_width)*2,false);
            expert_loan_->add(a->pooled,(a->maximum+a->pool-1)/a->pool,std::size_t(a->index_width)*2,false,a->pool);
            const auto& k=a->hot.exclusive()?a->hot.keys:a->keys;
            const auto& v=a->hot.exclusive()?a->hot.values:a->values;
            if(a->hot.exclusive() || (!a->host && !a->offload))for(int64_t h=0;h<a->heads;++h) {
                expert_loan_->add(k.narrow(1,h,1),k.size(2),std::size_t(a->key_width)*width_bytes(a->options),a->hot.exclusive());
                expert_loan_->add(v.narrow(1,h,1),v.size(2),std::size_t(a->value_width)*width_bytes(a->options),a->hot.exclusive());
            }
        }
    }
    if(offload_ && expert_loan_)offload_->set_expert_loan(expert_loan_);
    // Reserve shared decode I/O buffers before sizing the expert arenas.
    // Graph capture must not discover a large workspace after admission.
    const auto* stream_graphs=std::getenv("MFQ_KV_STREAM_GRAPHS");
    if(offload_ && (!stream_graphs || stream_graphs[0]!='0'))for(auto& weak:allocations_)if(auto a=weak.lock()) {
        if(a->selected_columns)offload_->gather_workspace(8*a->selected_columns,
            a->heads*(a->key_width+a->value_width),a->options.scalar_type(),a->heads,a->key_width);
    }
    ready_=true;
    gpu_budget_=index+std::min(raw,gpu);ram_required_=raw-std::min(raw,gpu);
    std::cerr<<"runtime_kv_memory placement="<<options_.placement<<" allocation="<<(options_.dynamic?"dynamic":"preallocated")
        <<" raw_bytes="<<raw<<" index_bytes="<<index<<" streaming="<<bool(offload_)<<'\n';
    if(automatic_ram_)std::cerr<<"runtime_kv_ram_resident cells="<<cells<<" maximum="<<maximum<<" raw_ram_bytes="<<raw<<" graph=1\n";
    if(tiered_budget)std::cerr<<"runtime_kv_tier_budget gpu_bytes="<<index+gpu<<" raw_ram_required="<<raw-std::min(raw,gpu)
        <<" ram_preallocated="<<options_.ram_preallocated<<" exclusive="<<exclusive<<"\n";
}
std::optional<std::size_t> KvMemoryManager::additional_preallocated_gpu_bytes(
        int64_t maximum, int64_t heads, std::size_t key_row_bytes, std::size_t value_row_bytes,
        int64_t index_width, int64_t pool) const {
    if (!ready_ || options_.dynamic) return std::nullopt;
    for (const auto& weak : allocations_) {
        const auto allocation = weak.lock();
        if (!allocation || allocation->maximum != maximum || allocation->heads != heads ||
            allocation->key_width * width_bytes(allocation->options) != key_row_bytes ||
            allocation->value_width * width_bytes(allocation->options) != value_row_bytes ||
            allocation->index_width != index_width || allocation->pool != pool) continue;
        if (!automatic_ram_ && !options_.budget_policy &&
            (options_.placement == "auto" || options_.placement == "auto_ssd"))
            return allocation->raw_bytes() + allocation->index_bytes();
        auto bytes = allocation->gpu_bytes();
        if (allocation->offload) bytes += allocation->offload->config().gpu_budget_bytes;
        return bytes;
    }
    return std::nullopt;
}

std::size_t KvMemoryManager::growth_bytes(int64_t tokens) const {
    std::size_t growth = 0;
    for (const auto& weak : allocations_)
        if (const auto allocation = weak.lock())
            growth += allocation->growth_bytes(std::min(allocation->maximum, tokens));
    return growth;
}
void KvMemoryManager::ensure(int64_t tokens) {
    if(expert_loan_)expert_loan_->ensure(tokens);
    const auto growth = growth_bytes(tokens);
    if(growth && reserve_)reserve_(growth+(64ULL<<20));
    for(auto& weak:allocations_)if(auto a=weak.lock())a->ensure(std::min(a->maximum,tokens));
}
std::size_t KvMemoryManager::growth_reserve() const {
    std::size_t result=0;
    for(auto& weak:allocations_)if(auto a=weak.lock())result+=a->growth_bytes(a->maximum);
    return result;
}
std::size_t KvMemoryManager::expert_budget_charge() const {
    std::size_t bytes=growth_reserve();
    for(auto& weak:allocations_)if(auto a=weak.lock())bytes+=a->gpu_bytes();
    if(offload_)bytes+=offload_->config().gpu_budget_bytes+
        (offload_->graph_workspace_bytes()>offload_->config().buffer_bytes?offload_->graph_workspace_bytes()-offload_->config().buffer_bytes:0);
    return bytes;
}
std::vector<std::pair<std::string,double>> KvMemoryManager::metrics() const {
    std::size_t gpu=0,ram=0,capacity=0,vmm=0;
    int64_t resident_cells=0;
    for(auto& weak:allocations_)if(auto a=weak.lock()){
        gpu+=a->gpu_bytes();ram+=a->ram_bytes();capacity+=a->raw_bytes()+a->index_bytes();
        if(a->virtual_keys)vmm+=a->gpu_bytes()-a->index_bytes();
        if(a->hot.defined())resident_cells=resident_cells?std::min(resident_cells,a->hot.keys.size(2)):a->hot.keys.size(2);
    }
    return {{"kv_memory_supported",capacity?1.0:0.0},{"kv_memory_dynamic",options_.dynamic?1.0:0.0},
        {"kv_memory_preallocated",options_.dynamic?0.0:1.0},{"kv_memory_capacity_bytes",double(capacity)},
        {"kv_memory_gpu_bytes",double(gpu+(offload_?offload_->expert_loan_bytes():0)-(expert_loan_?expert_loan_->bytes()-expert_loan_->external_bytes():0))},{"kv_memory_ram_bytes",double(ram)},
        {"kv_memory_expert_loan_bytes",double(expert_loan_?expert_loan_->bytes()-expert_loan_->external_bytes():0)},
        {"kv_memory_expert_loan_hit_bytes",double(expert_loan_?expert_loan_->hits():0)},
        {"kv_memory_expert_loan_lookup_us",expert_loan_?expert_loan_->lookup_us():0},
        {"kv_memory_gpu_budget_bytes",double(gpu_budget_)},{"kv_memory_cold_required_bytes",double(ram_required_)},
        {"kv_memory_ram_preallocated",options_.ram_preallocated?1.0:0.0},
        {"kv_memory_vmm_bytes",double(vmm)}, {"kv_memory_auto_ram",automatic_ram_?1.0:0.0},
        {"kv_memory_resident_cells",double(resident_cells)},
        {"kv_memory_ssd_enabled",options_.placement=="auto_ssd"?1.0:0.0}};
}
}
