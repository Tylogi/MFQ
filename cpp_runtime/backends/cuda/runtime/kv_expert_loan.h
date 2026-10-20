#pragma once
#include "mfq_tensor_backend.h"
#include "mfq_cuda_context.h"
#include "dma_copy_batch.h"
#include <array>
#include <list>
#include <map>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <unordered_map>
#include <functional>
#include <memory>
#include <limits>
#include <utility>

namespace mfq::cuda {
// Cache immutable expert fields in pages of KV storage that have never held
// live KV. Ordinary consumers copy into fixed decode staging addresses. The
// phased decode window may read a pinned field directly until the following
// route publication; next_packet() is called only after that reader boundary.
// KV growth is prepared before a window or at its ordered KV service boundary.
class KvExpertLoan {
    struct Entry;
    struct ProjectionLayout {
        std::array<std::size_t,8> bytes{},offsets{};
        int fields=0;std::size_t logical_bytes=0;
    };
public:
    void set_reuse_learning(bool enabled) {
        std::lock_guard lock(mutex_);reuse_learning_=enabled;
    }
    using Key=std::array<int64_t,3>; // source, expert, field
    struct ReuseStats {uint64_t promotions=0,protected_hits=0,probation_evictions=0,forecast_eviction_skips=0,protected_trims=0;
        uint64_t protected_accesses=0,accepted_row_promotions=0;
        std::size_t protected_pages=0,capacity_pages=0,protected_limit_pages=0;bool enabled=false;};
    void set_reuse_priority(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);if(reuse_priority_==enabled)return;
        reuse_priority_=enabled;reuse_pages_=0;reuse_heads_={};reuse_tails_={};
        for(auto* entries:{&lru_,&retired_})for(auto& entry:*entries) {
            entry.reused=false;entry.reuse_linked=false;entry.reuse_prev=entry.reuse_next=nullptr;
            if(entry.published)notify_projection(entry,true,true);
        }
        if(enabled)rebuild_reuse_queues();
    }
    ReuseStats reuse_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);auto result=reuse_stats_;result.protected_pages=reuse_pages_;
        result.capacity_pages=reuse_capacity_pages_;result.protected_limit_pages=reuse_capacity_pages_/2;
        result.enabled=reuse_priority_;return result;
    }
    class ExchangeReadGuard {
        friend class KvExpertLoan;
        KvExpertLoan* owner_;
        bool all_;
        std::vector<Entry*> pins_;
        explicit ExchangeReadGuard(KvExpertLoan* owner,bool all,std::size_t reserve):owner_(owner),all_(all) {
            pins_.reserve(reserve);
        }
    public:
        ExchangeReadGuard(const ExchangeReadGuard&)=delete;
        ExchangeReadGuard(ExchangeReadGuard&& other) noexcept:owner_(std::exchange(other.owner_,nullptr)),
            all_(other.all_),pins_(std::move(other.pins_)){}
        ~ExchangeReadGuard(){if(owner_)owner_->release_exchange_reads(*this);}
    };
    // Concurrent predictor packets may reclaim unrelated pages. Only addresses
    // retained by an exchange (or an explicit whole-cache guard) stay pinned.
    ExchangeReadGuard retain_exchange_reads(bool all=true,std::size_t reserve=0) {
        std::lock_guard<std::mutex> lock(mutex_);
        ExchangeReadGuard guard(this,all,reserve);
        ++exchange_readers_;exchange_freezes_+=all;
        return guard;
    }
    struct SpeculativeStats {uint64_t windows=0,kept=0,reverted_hits=0,cold_insertions=0,cancels=0,disabled=0;};
    static constexpr std::size_t page=64<<10;
    void begin_speculative(int rows) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(speculative_rows_)throw std::logic_error("nested KV expert speculation");
        if(rows<1 || rows>32)throw std::invalid_argument("invalid KV expert speculative rows");
        speculative_keys_.clear();packet_sources_={};packet_masks_.clear();
        try {speculative_keys_.reserve(std::max(speculative_keys_.capacity(),lru_.size()+64));}
        catch(const std::bad_alloc&){++speculative_stats_.disabled;return;}
        if(++speculative_epoch_==0) {
            for(auto* entries:{&lru_,&retired_})for(auto& entry:*entries)entry.epoch=0;
            speculative_epoch_=1;
        }
        // Assign the current order once. Ordinary cache use needs no timestamp
        // maintenance; speculative execution keeps the original live LRU.
        access_order_=0;
        for(auto it=lru_.rbegin();it!=lru_.rend();++it)it->order=++access_order_;
        speculative_rows_=rows;
    }
    void speculative_packet(const std::array<int64_t,3>& sources,const int32_t* ids,int rows,int routes,int experts) {
        std::lock_guard<std::mutex> lock(mutex_);
        speculative_packet_locked(sources,ids,rows,routes,experts);
    }
    void commit_speculative(int rows) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!speculative_rows_)return;
        if(rows<0 || rows>speculative_rows_)throw std::invalid_argument("invalid KV expert committed prefix");
        finish_speculative(rows,false);
    }
    void discard_speculative() noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if(speculative_rows_)finish_speculative(0,true);
        }catch(...){}
    }
    SpeculativeStats speculative_stats() const {std::lock_guard<std::mutex> lock(mutex_);return speculative_stats_;}
    std::size_t add(mfq_tensor_backend::Tensor owner,int64_t capacity,std::size_t row_bytes,bool prefix,int64_t ratio=1,bool manual=false,bool external=false) {
        std::lock_guard<std::mutex> lock(mutex_);
        Region region;region.owner=std::move(owner);region.capacity=capacity;region.row_bytes=row_bytes;region.prefix=prefix;region.ratio=ratio;
        region.manual_live=manual?0:-1;
        region.external=external;
        region.end=region.owner.nbytes()/page;region.largest_free=region.end;if(region.end)region.free[0]=region.end;
        std::size_t index=0;
        while(index<regions_.size() && regions_[index].owner.defined())++index;
        if(index==regions_.size())regions_.push_back(std::move(region));
        else regions_[index]=std::move(region);
        update_bounds();
        initialized_.record(current_stream().stream());
        initialization_waited_=false;
        return index;
    }
    // Called at a completed request/window boundary. Revoke published expert
    // addresses and drain their uploads before returning this allocation.
    void remove(std::size_t region) {
        std::unique_lock<std::mutex> lock(mutex_);
        exchange_finished_.wait(lock,[&]{return !exchange_readers_;});
        auto& r=regions_.at(region);
        for(auto* entries:{&lru_,&retired_})for(auto it=entries->begin();it!=entries->end();) {
            if(it->region==region){auto old=it++;erase(old);}else ++it;
        }
        if(copy_stream_)MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
        r=Region{};
        update_bounds();
    }
    void ensure(int64_t tokens) {
        std::unique_lock<std::mutex> lock(mutex_);
        if(tokens<=high_water_)return;high_water_=tokens;
        if(exchange_readers_ && std::any_of(regions_.begin(),regions_.end(),[&](const Region& r) {
                return r.owner.defined() && bounds(r,tokens)!=std::pair{r.begin,r.end};
            }))exchange_finished_.wait(lock,[&]{return !exchange_readers_;});
        // Most token appends stay inside the same 64 KiB protection pages.
        // Only changed page boundaries can invalidate a cached expert.
        if(update_bounds())revoke();
    }
    void protect(std::size_t region,int64_t rows) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto& r=regions_.at(region);if(rows<=r.manual_live)return;
        if(exchange_readers_ && bounds(r,high_water_,rows)!=std::pair{r.begin,r.end})
            exchange_finished_.wait(lock,[&]{return !exchange_readers_;});
        r.manual_live=rows;if(update_bounds())revoke();
    }
    void next_packet(const std::array<int64_t,3>& sources={},const int32_t* ids=nullptr,int rows=0,int routes=0,int experts=0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(++packet_==0){for(auto& e:lru_){e.packet=0;e.forecast_packet=0;e.last_use_packet=0;}packet_=1;}
        // The next route publication fences readers of the previous packet,
        // exactly as for ordinary LRU eviction. Retirement itself never frees
        // an address that an already submitted graph can still be reading.
        if(!exchange_freezes_)for(auto it=retired_.begin();it!=retired_.end();) {
            if(it->exchange_pins){++it;continue;}
            retired_bytes_+=it->bytes;auto old=it++;erase(old);
        }
        // The previous route's consumers have finished before this boundary.
        // Publish only now: an allocated projection may still have DMA pending
        // during its first use and must not enter the next prepare's hot set.
        for(const auto& key:pending_publications_)if(auto found=entries_.find(key);found!=entries_.end()) {
            auto& entry=*found->second;
            if(!entry.retired && !entry.published){entry.published=true;notify_projection(entry,true);}
        }
        pending_publications_.clear();
        if(ids)speculative_packet_locked(sources,ids,rows,routes,experts);
    }
    // Pin every known member of this packet before an allocation can evict a
    // later routed projection. Actual pin() calls still update LRU recency.
    std::size_t protect_packet_fields(const std::vector<Key>& keys,bool forecast=false) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t protected_fields=0;
        for(const auto& key:keys)if(auto found=entries_.find(key);found!=entries_.end())
            {found->second->packet=packet_;if(forecast)found->second->forecast_packet=packet_;++protected_fields;}
        return protected_fields;
    }
    using ProjectionFill=std::function<void(void*,const std::array<std::size_t,8>&)>;
    using ProjectionObserver=std::function<void(int64_t,const void*,const std::array<std::size_t,8>&)>;
    // RAM eviction may use only repeatedly consumed GPU images. One-use
    // transfer pages are too short-lived to replace a protected RAM image.
    void observe_projections(int64_t source,ProjectionObserver observer,std::shared_ptr<void> owner={},bool reuse_only=false) {
        if(!observer)throw std::invalid_argument("empty KV projection observer");
        std::lock_guard<std::mutex> lock(mutex_);
        for(const auto& entry:lru_)if(entry.key[0]==source && entry.projection && entry.published)
            observer(entry.key[1],!reuse_only || entry.reused?pointer(entry):nullptr,entry.projection->offsets);
        auto& observers=observers_[source];
        std::erase_if(observers,[](const auto& value){return value.managed && value.owner.expired();});
        observers.push_back({std::move(observer),owner,bool(owner),reuse_only});
    }
    // Route publication has fenced the previous readers. Check complete hits
    // before acquiring RAM expert slices or reserving transfer-stage slots.
    const void* pin_existing_projection(int64_t source,int64_t expert,const std::vector<int64_t>& sizes,
            std::array<std::size_t,8>& offsets,std::size_t& logical_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found=entries_.find({source,expert,-1});if(found==entries_.end())return nullptr;
        auto it=found->second;
        if(!it->projection || sizes.size()!=std::size_t(it->projection->fields))
            throw std::logic_error("KV existing projection field count changed");
        for(std::size_t f=0;f<sizes.size();++f)
            if(sizes[f]<0 || std::size_t(sizes[f])!=it->projection->bytes[f])
                throw std::logic_error("KV existing projection field geometry changed");
        if(it->retired)pending_publications_.push_back(it->key);
        note_access(*it,it->retired);
        lru_.splice(lru_.begin(),it->retired?retired_:lru_,it);it->retired=false;it->packet=packet_;
        if(reuse_priority_)reuse_touch(*it);
        offsets=it->projection->offsets;logical_bytes=it->projection->logical_bytes;hits_+=logical_bytes;
        return pointer(*it);
    }
    const void* pin_projection(int64_t source,int64_t expert,const std::vector<int64_t>& sizes,
            cudaStream_t stream,bool& hit,const ProjectionFill& populate) {
        if(sizes.empty() || sizes.size()>8 || !populate)
            throw std::invalid_argument("invalid KV projection layout");
        ProjectionLayout layout;layout.fields=int(sizes.size());std::size_t total=0;
        for(std::size_t f=0;f<sizes.size();++f) {
            if(sizes[f]<0 || total>std::numeric_limits<std::size_t>::max()-15)
                throw std::overflow_error("KV projection field size overflow");
            total=(total+15)&~std::size_t(15);layout.offsets[f]=total;layout.bytes[f]=std::size_t(sizes[f]);
            if(layout.bytes[f]>std::numeric_limits<std::size_t>::max()-total)
                throw std::overflow_error("KV projection payload overflow");
            total+=layout.bytes[f];layout.logical_bytes+=layout.bytes[f];
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return pin_locked({source,expert,-1},nullptr,total,stream,hit,nullptr,{},&layout,populate);
    }
    void retire_expert(int64_t source,int64_t expert,int fields) {
        std::lock_guard<std::mutex> lock(mutex_);
        for(int f=-1;f<fields;++f)if(auto found=entries_.find({source,expert,f});found!=entries_.end()) {
            auto it=found->second;if(it->retired)continue;
            set_reused(*it,false);
            reuse_unlink(*it);
            if(it->published){notify_projection(*it,false);it->published=false;}
            it->retired=true;retired_.splice(retired_.end(),lru_,it);
        }
    }
    // Residency exchange holds the completed window and joins before the next
    // packet/KV growth. It may reuse these bytes until that exchange finishes.
    bool contains_projection(int64_t source,int64_t expert,const std::vector<int64_t>& sizes) const {
        std::lock_guard lock(mutex_);
        const auto found=entries_.find({source,expert,-1});
        if(found==entries_.end() || !found->second->projection)return false;
        const auto& layout=*found->second->projection;
        if(sizes.size()!=std::size_t(layout.fields))return false;
        for(std::size_t f=0;f<sizes.size();++f)
            if(sizes[f]<0 || std::size_t(sizes[f])!=layout.bytes[f])return false;
        return true;
    }
    const void* find_for_exchange(const Key& key,std::size_t bytes,ExchangeReadGuard* guard=nullptr) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto retain=[&](Entry& entry) {
            if(!guard)return;
            if(guard->owner_!=this)throw std::logic_error("KV exchange guard belongs to another cache");
            if(guard->all_)return;
            // Complete projection fields are adjacent lookups. A repeated
            // nonadjacent lookup may hold another pin; release balances both.
            if(guard->pins_.empty() || guard->pins_.back()!=&entry) {
                guard->pins_.push_back(&entry);++entry.exchange_pins;
            }
        };
        const auto found=entries_.find(key);
        if(found!=entries_.end()) {
            auto& entry=*found->second;
            if(entry.bytes!=bytes)throw std::logic_error("KV exchange field geometry changed");
            retain(entry);
            return pointer(entry);
        }
        const auto projection=entries_.find({key[0],key[1],-1});
        if(projection==entries_.end())return nullptr;
        auto& entry=*projection->second;
        if(!entry.projection || key[2]<0 || key[2]>=entry.projection->fields ||
                entry.projection->bytes[std::size_t(key[2])]!=bytes)
            throw std::logic_error("KV exchange projection field geometry changed");
        retain(entry);
        return static_cast<const uint8_t*>(pointer(entry))+entry.projection->offsets[std::size_t(key[2])];
    }
    const void* pin(const Key& key,const void* source,std::size_t bytes,cudaStream_t stream,bool& hit,std::vector<DmaCopy>* batch=nullptr,
                    const std::function<const void*()>& stage_source={}) {
        std::lock_guard<std::mutex> lock(mutex_);return pin_locked(key,source,bytes,stream,hit,batch,stage_source);
    }
    bool copy(const Key& key,const void* source,void* destination,std::size_t bytes,cudaStream_t stream,bool& hit) {
        std::lock_guard<std::mutex> lock(mutex_);const auto* cache=pin_locked(key,source,bytes,stream,hit);
        if(cache)MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(destination,cache,bytes,cudaMemcpyDeviceToDevice,stream));
        return cache!=nullptr;
    }
    std::size_t bytes() const {std::lock_guard<std::mutex> lock(mutex_);return bytes_;}
    std::size_t external_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);std::size_t result=0;
        for(const auto* entries:{&lru_,&retired_})for(const auto& entry:*entries)
            if(regions_[entry.region].external)result+=entry.bytes;
        return result;
    }
    std::size_t hits() const {std::lock_guard<std::mutex> lock(mutex_);return hits_;}
    std::size_t retired_bytes() const {std::lock_guard<std::mutex> lock(mutex_);return retired_bytes_;}
    double lookup_us() const {std::lock_guard<std::mutex> lock(mutex_);return lookup_us_;}
    std::size_t region_bytes(std::size_t region) const {
        std::lock_guard<std::mutex> lock(mutex_);std::size_t result=0;
        for(const auto* entries:{&lru_,&retired_})for(const auto& e:*entries)if(e.region==region)result+=e.bytes;return result;
    }
private:
    void speculative_packet_locked(const std::array<int64_t,3>& sources,const int32_t* ids,int rows,int routes,int experts) {
        if(!speculative_rows_)return;
        if(rows!=speculative_rows_ || !ids || routes<=0 || experts<=0) {disable_speculative();return;}
        try {packet_masks_.assign(experts,0);}
        catch(const std::bad_alloc&){disable_speculative();return;}
        packet_sources_=sources;
        for(int row=0;row<rows;++row)for(int route=0;route<routes;++route) {
            const auto expert=ids[row*routes+route];
            if(unsigned(expert)<unsigned(experts))packet_masks_[expert]|=uint32_t(1)<<row;
        }
    }
    struct Region {mfq_tensor_backend::Tensor owner;int64_t capacity=0,ratio=1,manual_live=-1;std::size_t row_bytes=0,begin=0,end=0,largest_free=0;bool prefix=false,external=false;std::map<std::size_t,std::size_t> free;};
    struct Entry {Key key;std::size_t region,first,pages,bytes;uint64_t packet=0;bool retired=false,published=false;
        uint64_t forecast_packet=0;
        uint64_t order=0,prior_order=0,epoch=0,last_use_packet=0,prior_use_packet=0;
        uint32_t used_rows=0;bool reused=false,reuse_eligible=false,reuse_linked=false;
        Entry* reuse_prev=nullptr;Entry* reuse_next=nullptr;
        std::unique_ptr<ProjectionLayout> projection;std::size_t exchange_pins=0;};
    void disable_speculative() {
        speculative_rows_=0;speculative_keys_.clear();packet_sources_={};packet_masks_.clear();++speculative_stats_.disabled;
    }
    void note_access(Entry& entry,bool fresh=false) {
        if(!reuse_learning_)return;
        // Eviction protection is not a weight access. In particular, packet
        // preflight must not hide a later genuine hit from reuse learning.
        const bool reused=!fresh && entry.last_use_packet && entry.last_use_packet!=packet_;
        if(reuse_priority_ && reused && entry.packet==packet_ && !reuse_stats_.protected_accesses++)
            std::fprintf(stderr,"moe_kv_real_reuse scope=first_protected_hit enabled=1\n");
        if(reuse_priority_ && entry.reused)++reuse_stats_.protected_hits;
        if(!speculative_rows_) {
            entry.last_use_packet=packet_;
            if(reuse_priority_ && reused){set_reused(entry,true);trim_reused(&entry);}
            return;
        }
        const auto expert=entry.key[1];
        const auto mask=expert>=0 && std::size_t(expert)<packet_masks_.size() &&
            std::find(packet_sources_.begin(),packet_sources_.end(),entry.key[0])!=packet_sources_.end()?
                packet_masks_[std::size_t(expert)]:0;
        if(mask) {
            if(entry.epoch!=speculative_epoch_) {
                try {speculative_keys_.push_back(entry.key);}
                catch(const std::bad_alloc&){disable_speculative();return;}
                entry.epoch=speculative_epoch_;entry.prior_order=fresh?0:entry.order;entry.used_rows=0;
                entry.prior_use_packet=fresh?0:entry.last_use_packet;
                entry.reuse_eligible=reused;
            }
            entry.used_rows|=mask;
        }else if(reuse_priority_ && reused){set_reused(entry,true);trim_reused(&entry);}
        entry.last_use_packet=packet_;entry.order=++access_order_;
    }
    void finish_speculative(int rows,bool cancelled) {
        const uint32_t accepted=rows==32?UINT32_MAX:(uint32_t(1)<<rows)-1;
        const auto token_stride=access_order_+1;
        bool reordered=false,token_ordered=false;uint64_t hits=0,insertions=0;
        for(const auto& key:speculative_keys_)if(auto found=entries_.find(key);found!=entries_.end()) {
            auto& entry=*found->second;if(entry.epoch!=speculative_epoch_)continue;
            entry.epoch=0;
            const auto used=entry.used_rows&accepted;
            if(used) {
                const bool repeated_rows=(used&(used-1))!=0;
                if(reuse_priority_ && (entry.reuse_eligible || repeated_rows)) {
                    if(!entry.reused && !entry.reuse_eligible && repeated_rows && !reuse_stats_.accepted_row_promotions++)
                        std::fprintf(stderr,"moe_kv_accepted_row_reuse scope=first_promotion enabled=1\n");
                    set_reused(entry,true);
                }
                if(rows>1) {
                    // Batched GPU consumption runs layer by layer. Match the
                    // accepted token chronology used by RAM learning instead
                    // of letting read completion order rank the next token's
                    // expert cache. Rejected later rows never affect this rank.
                    unsigned last_row=0;for(auto mask=used>>1;mask;mask>>=1)++last_row;
                    entry.order=(last_row+1)*token_stride+entry.order;
                    reordered=token_ordered=true;
                }
                ++speculative_stats_.kept;continue;
            }
            entry.order=entry.prior_order;entry.last_use_packet=entry.prior_use_packet;reordered=true;
            if(entry.prior_order)++hits;else ++insertions;
        }
        // Rejected hits recover their pre-window recency. New rejected entries
        // become cold. Do not free, unpublish or move any payload here: a later
        // ordinary packet boundary remains the sole eviction/read fence.
        if(reordered)lru_.sort([](const Entry& a,const Entry& b){return a.order>b.order;});
        if(reuse_priority_){rebuild_reuse_queues();trim_reused();}
        if(token_ordered && !token_order_reported_) {
            token_order_reported_=true;
            std::fprintf(stderr,"moe_kv_token_order scope=first_committed_window enabled=1 rows=%d\n",rows);
        }
        ++speculative_stats_.windows;speculative_stats_.reverted_hits+=hits;
        speculative_stats_.cold_insertions+=insertions;speculative_stats_.cancels+=cancelled;
        if(reordered && !speculative_reported_) {
            speculative_reported_=true;
            std::fprintf(stderr,"moe_kv_mtp_lru_dispatch enabled=1 rows=%d window_rows=%d reverted_hits=%llu cold_insertions=%llu\n",
                rows,speculative_rows_,static_cast<unsigned long long>(hits),static_cast<unsigned long long>(insertions));
        }
        speculative_rows_=0;speculative_keys_.clear();packet_sources_={};packet_masks_.clear();
    }
    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            std::size_t hash=0;
            for(const auto value:key)hash^=std::hash<int64_t>{}(value)+0x9e3779b9+(hash<<6)+(hash>>2);
            return hash;
        }
    };
    const void* pin_locked(const Key& key,const void* source,std::size_t bytes,cudaStream_t stream,bool& hit,std::vector<DmaCopy>* batch=nullptr,
                           const std::function<const void*()>& stage_source={},
                           const ProjectionLayout* projection=nullptr,const ProjectionFill& populate={}) {
        const auto started=std::chrono::steady_clock::now();
        struct Timer {double& total;std::chrono::steady_clock::time_point start;
            ~Timer(){total+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();}} timer{lookup_us_,started};
        hit=false;if(bytes<page)return nullptr;
        if(auto found=entries_.find(key);found!=entries_.end()) {
            auto it=found->second;
            if(it->bytes!=bytes)throw std::logic_error("KV expert loan field geometry changed");
            if(projection && (!it->projection || projection->fields!=it->projection->fields ||
                    projection->bytes!=it->projection->bytes))
                throw std::logic_error("KV expert projection geometry changed");
            if(it->retired && it->projection)pending_publications_.push_back(it->key);
            note_access(*it,it->retired);
            lru_.splice(lru_.begin(),it->retired?retired_:lru_,it);it->retired=false;
            if(reuse_priority_)reuse_touch(*it);
            it->packet=packet_;hit=true;hits_+=projection?projection->logical_bytes:bytes;
            return pointer(*it);
        }
        const auto pages=(bytes+page-1)/page;
        if(std::none_of(regions_.begin(),regions_.end(),[&](const Region& r){return r.end>=r.begin && r.end-r.begin>=pages;}))return nullptr;
        auto room=find(pages);
        if(room.first==regions_.size() && exchange_freezes_)return nullptr;
        const auto replaceable=[&](Entry& entry) {
            if(entry.exchange_pins)return false;
            if(entry.packet!=packet_)return true;
            if(entry.forecast_packet==packet_)++reuse_stats_.forecast_eviction_skips;
            return false;
        };
        while(room.first==regions_.size()) {
            auto candidate=lru_.end();
            if(reuse_priority_) {
                for(int queue=0;queue<2 && candidate==lru_.end();++queue)
                    for(auto* entry=reuse_tails_[queue];entry;entry=entry->reuse_prev)
                        if(replaceable(*entry)){candidate=entries_.at(entry->key);break;}
            }else for(auto it=lru_.end();it!=lru_.begin();) {
                --it;if(replaceable(*it)){candidate=it;break;}
            }
            if(candidate==lru_.end())break;
            if(reuse_priority_ && !candidate->reused)++reuse_stats_.probation_evictions;
            erase(candidate);room=find(pages);
        }
        if(room.first==regions_.size())return nullptr;
        // Pageable experts need a pinned upload only on a cache miss. Hits
        // must not copy their weights through CPU staging again.
        if(stage_source)source=stage_source();
        Entry entry{key,room.first,room.second,pages,bytes,packet_};
        if(projection)entry.projection=std::make_unique<ProjectionLayout>(*projection);
        auto* cache=pointer(entry);
        if(!initialization_waited_ || copy_stream_!=stream) {
            MFQ_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(stream,initialized_.get(),0));
            initialization_waited_=true;
        }
        if(projection)populate(cache,projection->offsets);
        else if(batch)batch->push_back({cache,source,bytes});
        else MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(cache,source,bytes,cudaMemcpyHostToDevice,stream));
        auto& r=regions_[room.first];const auto previous=r.free.at(entry.first),remaining=previous-pages;
        r.free.erase(entry.first);if(remaining)r.free.emplace(entry.first+pages,remaining);
        if(previous==r.largest_free) {
            r.largest_free=0;for(const auto& [begin,size]:r.free)r.largest_free=std::max(r.largest_free,size);
        }
        note_access(entry,true);
        lru_.push_front(std::move(entry));entries_[key]=lru_.begin();bytes_+=bytes;copy_stream_=stream;
        if(reuse_priority_)reuse_touch(lru_.front());
        if(projection)pending_publications_.push_back(key);
        return cache;
    }

    static std::pair<std::size_t,std::size_t> bounds(const Region& r,int64_t high_water,int64_t manual=-2) {
        const auto protected_rows=manual==-2?r.manual_live:manual;
        const auto live=std::min(r.capacity,protected_rows>=0?protected_rows:(high_water+r.ratio-1)/r.ratio);
        return {r.prefix?0:(std::size_t(live)*r.row_bytes+page-1)/page,
            r.prefix?std::size_t(r.capacity-live)*r.row_bytes/page:r.owner.nbytes()/page};
    }
    void release_exchange_reads(const ExchangeReadGuard& guard) noexcept {
        {std::lock_guard<std::mutex> lock(mutex_);
            for(auto* entry:guard.pins_)--entry->exchange_pins;
            --exchange_readers_;exchange_freezes_-=guard.all_;}
        exchange_finished_.notify_all();
    }
    bool update_bounds() {
        bool changed=false;
        for(auto& r:regions_) {
            if(!r.owner.defined())continue;
            const auto [begin,end]=bounds(r,high_water_);
            if(begin==r.begin && end==r.end)continue;
            r.begin=begin;r.end=end;changed=true;
            for(auto it=r.free.begin();it!=r.free.end();) {
                const auto begin=std::max(it->first,r.begin),end=std::min(it->first+it->second,r.end);
                if(begin==it->first && end==it->first+it->second){++it;continue;}
                it=r.free.erase(it);if(end>begin)r.free.emplace(begin,end-begin);
            }
            r.largest_free=0;for(const auto& [begin,size]:r.free)r.largest_free=std::max(r.largest_free,size);
        }
        reuse_capacity_pages_=0;
        for(const auto& r:regions_)if(r.end>r.begin)reuse_capacity_pages_+=r.end-r.begin;
        if(reuse_priority_)trim_reused();
        return changed;
    }
    void set_reused(Entry& entry,bool value) {
        if(entry.reused==value)return;
        reuse_unlink(entry);
        entry.reused=value;
        if(entry.published)notify_projection(entry,true,true);
        if(value){reuse_pages_+=entry.pages;++reuse_stats_.promotions;}
        else reuse_pages_-=entry.pages;
        if(reuse_priority_ && !entry.retired)reuse_touch(entry);
    }
    void reuse_unlink(Entry& entry) {
        if(!entry.reuse_linked)return;
        const int queue=entry.reused?1:0;
        if(entry.reuse_prev)entry.reuse_prev->reuse_next=entry.reuse_next;
        else reuse_heads_[queue]=entry.reuse_next;
        if(entry.reuse_next)entry.reuse_next->reuse_prev=entry.reuse_prev;
        else reuse_tails_[queue]=entry.reuse_prev;
        entry.reuse_prev=entry.reuse_next=nullptr;entry.reuse_linked=false;
    }
    void reuse_touch(Entry& entry) {
        reuse_unlink(entry);const int queue=entry.reused?1:0;
        entry.reuse_next=reuse_heads_[queue];
        if(entry.reuse_next)entry.reuse_next->reuse_prev=&entry;
        else reuse_tails_[queue]=&entry;
        reuse_heads_[queue]=&entry;entry.reuse_linked=true;
    }
    void rebuild_reuse_queues() {
        reuse_heads_={};reuse_tails_={};
        for(auto it=lru_.rbegin();it!=lru_.rend();++it) {
            it->reuse_linked=false;it->reuse_prev=it->reuse_next=nullptr;reuse_touch(*it);
        }
    }
    void trim_reused(Entry* recent=nullptr) {
        // A quarter can be smaller than one token's streamed working set.
        // Reserve half for new routes so they can survive until their next
        // use; the other half still protects recurring experts from scans.
        const auto limit=reuse_capacity_pages_/2;
        const auto demote=[&](Entry& entry) {
            set_reused(entry,false);
            if(!reuse_stats_.protected_trims++)std::fprintf(stderr,
                "moe_kv_reuse_budget scope=first_trim enabled=1 protected_fraction=0.5 capacity_pages=%zu\n",
                reuse_capacity_pages_);
        };
        while(reuse_pages_>limit) {
            auto* entry=reuse_tails_[1];
            if(entry && entry==recent)entry=entry->reuse_prev;
            if(!entry)break;
            demote(*entry);
        }
        if(reuse_pages_>limit && recent)demote(*recent);
    }
    void revoke() {
        bool revoked=false;
        for(auto* entries:{&lru_,&retired_})for(auto it=entries->begin();it!=entries->end();) {
            const auto& r=regions_[it->region];
            if(it->first<r.begin || it->first+it->pages>r.end){auto old=it++;erase(old);revoked=true;}else ++it;
        }
        if(revoked && copy_stream_) {
            released_.record(copy_stream_);
            MFQ_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(current_stream().stream(),released_.get(),0));
        }
    }
    std::pair<std::size_t,std::size_t> find(std::size_t pages) {
        // Keep large holes available for large projections. Choosing the first
        // fitting address can evict live entries even when an exact fit exists
        // in another borrowed region. All addresses remain fixed after pinning.
        std::pair<std::size_t,std::size_t> best{regions_.size(),0};
        std::size_t first_size=0,best_size=std::numeric_limits<std::size_t>::max();
        for(std::size_t i=0;i<regions_.size();++i) {
            if(regions_[i].largest_free<pages)continue;
            for(const auto& [begin,size]:regions_[i].free) {
                if(size<pages || size>=best_size)continue;
                if(!first_size)first_size=size;
                best={i,begin};best_size=size;
                if(size==pages)break;
            }
            if(best_size==pages)break;
        }
        if(best_size<first_size && !best_fit_reported_) {
            best_fit_reported_=true;
            std::fprintf(stderr,"moe_kv_best_fit scope=first_tighter_fit enabled=1 requested_pages=%zu first_fit_pages=%zu selected_pages=%zu\n",
                pages,first_size,best_size);
        }
        return best;
    }
    void* pointer(const Entry& e) const {return static_cast<uint8_t*>(regions_[e.region].owner.data_ptr())+e.first*page;}
    void notify_projection(const Entry& entry,bool valid,bool reuse_change=false) {
        if(!entry.projection)return;
        if(auto found=observers_.find(entry.key[0]);found!=observers_.end()) {
            std::erase_if(found->second,[](const auto& value){return value.managed && value.owner.expired();});
            for(const auto& observer:found->second)if(!reuse_change || observer.reuse_only)
                observer.callback(entry.key[1],valid && (!observer.reuse_only || entry.reused)?pointer(entry):nullptr,entry.projection->offsets);
        }
    }
    void erase(std::list<Entry>::iterator it) {
        if(it->exchange_pins)throw std::logic_error("reclaiming a pinned KV exchange source");
        if(it->published)notify_projection(*it,false);
        set_reused(*it,false);
        reuse_unlink(*it);
        auto& r=regions_[it->region];
        auto begin=std::max(it->first,r.begin),end=std::min(it->first+it->pages,r.end);
        if(begin<end) {
            auto next=r.free.lower_bound(begin);
            if(next!=r.free.begin()) {auto prior=std::prev(next);if(prior->first+prior->second==begin){begin=prior->first;r.free.erase(prior);}}
            if(next!=r.free.end() && next->first==end){end=next->first+next->second;r.free.erase(next);}
            r.free.emplace(begin,end-begin);
            r.largest_free=std::max(r.largest_free,end-begin);
        }
        bytes_-=it->bytes;entries_.erase(it->key);
        (it->retired?retired_:lru_).erase(it);
    }
    mutable std::mutex mutex_;
    std::condition_variable exchange_finished_;
    std::size_t exchange_readers_=0,exchange_freezes_=0;
    std::vector<Region> regions_;
    std::list<Entry> lru_,retired_;
    std::unordered_map<Key,std::list<Entry>::iterator,KeyHash> entries_;
    struct Observer {ProjectionObserver callback;std::weak_ptr<void> owner;bool managed=false,reuse_only=false;};
    std::unordered_map<int64_t,std::vector<Observer>> observers_;
    std::vector<Key> pending_publications_;
    uint64_t packet_=1;
    int speculative_rows_=0;uint64_t speculative_epoch_=0,access_order_=0;
    std::array<int64_t,3> packet_sources_{};
    std::vector<uint32_t> packet_masks_;
    std::vector<Key> speculative_keys_;
    SpeculativeStats speculative_stats_;
    bool reuse_priority_=false,reuse_learning_=true;
    std::array<Entry*,2> reuse_heads_{},reuse_tails_{};
    std::size_t reuse_pages_=0,reuse_capacity_pages_=0;
    ReuseStats reuse_stats_;
    bool speculative_reported_=false,token_order_reported_=false;
    std::size_t bytes_=0,hits_=0,retired_bytes_=0;
    double lookup_us_=0;
    int64_t high_water_=0;
    cudaStream_t copy_stream_=nullptr;
    bool initialization_waited_=false,best_fit_reported_=false;
    Event initialized_,released_;
};
}
