#pragma once
#ifdef MFQ_NATIVE_CUDA_RUNTIME
#include "native/context.h"
#include "native/tensor/tensor.h"
#include <map>
#include <memory>
#include <mutex>
#include <set>

// One bounded pinned allocation; expert leases own independently reusable
// intervals. Returning an interval never allocates, including under pressure.
class MoePinnedExpertArena {
    using Free=std::map<std::size_t,std::size_t>;
    using Fits=std::set<std::pair<std::size_t,std::size_t>>;
    struct State {
        mfq::cuda::HostBuffer storage;
        std::mutex mutex;
        Free free;
        Fits fits;
        std::size_t used=0;
        explicit State(std::size_t bytes):storage(bytes,true) {
            free.emplace(0,bytes);fits.emplace(bytes,0);mfq::cuda::charge_tensor_host_bytes(bytes);
        }
        ~State(){mfq::cuda::tensor_host_bytes.fetch_sub(storage.size());}
    };
public:
    struct Block {
        std::shared_ptr<State> state;
        Free::node_type interval;
        Fits::node_type size_interval;
        ~Block() {
            if(!state)return;
            std::lock_guard lock(state->mutex);
            state->used-=interval.mapped();
            auto next=state->free.lower_bound(interval.key());
            if(next!=state->free.begin()) {
                const auto previous=std::prev(next);
                if(previous->first+previous->second==interval.key()) {
                    state->fits.erase({previous->second,previous->first});
                    interval.key()=previous->first;interval.mapped()+=previous->second;
                    state->free.erase(previous);
                }
            }
            if(next!=state->free.end() && interval.key()+interval.mapped()==next->first) {
                state->fits.erase({next->second,next->first});
                interval.mapped()+=next->second;state->free.erase(next);
            }
            size_interval.value()={interval.mapped(),interval.key()};
            state->free.insert(std::move(interval));state->fits.insert(std::move(size_interval));
        }
        std::uint8_t* data() const {return static_cast<std::uint8_t*>(state->storage.data())+interval.key();}
        std::size_t bytes() const {return interval.mapped();}
    };
    explicit MoePinnedExpertArena(std::size_t bytes):state_(std::make_shared<State>(bytes)) {}
    std::shared_ptr<Block> allocate(std::size_t bytes) {
        if(!bytes || bytes>SIZE_MAX-15)return {};
        bytes=(bytes+15)&~std::size_t(15);
        auto block=std::make_shared<Block>();
        Free node;node.emplace(0,bytes);block->interval=node.extract(node.begin());
        Fits size_node;size_node.emplace(bytes,0);block->size_interval=size_node.extract(size_node.begin());
        std::lock_guard lock(state_->mutex);
        // The size index avoids scanning fragmented intervals and preserves
        // larger holes for the next, larger expert format.
        if(auto fit=state_->fits.lower_bound({bytes,0});fit!=state_->fits.end()) {
            auto it=state_->free.find(fit->second);
            const auto offset=it->first;
            if(it->second==bytes) {state_->free.erase(it);state_->fits.erase(fit);}
            else {
                auto rest=state_->free.extract(it);rest.key()+=bytes;rest.mapped()-=bytes;
                auto size_rest=state_->fits.extract(fit);size_rest.value()={rest.mapped(),rest.key()};
                state_->free.insert(std::move(rest));state_->fits.insert(std::move(size_rest));
            }
            block->interval.key()=offset;block->state=state_;state_->used+=bytes;return block;
        }
        return {};
    }
    void* data() const {return state_->storage.data();}
    std::size_t bytes() const {return state_->storage.size();}
    std::size_t used_bytes() const {std::lock_guard lock(state_->mutex);return state_->used;}
    mfq::cuda::Tensor tensor() const {
        auto storage=std::make_shared<mfq::cuda::TensorStorage>();
        const auto device=mfq::cuda::Device{mfq::cuda::DeviceType::cpu,0};
        storage->owner=state_;storage->base=data();storage->bytes=bytes();storage->device=device;storage->pinned=true;
        const std::array<std::int64_t,1> shape{std::int64_t(bytes())};
        return mfq::cuda::Tensor(storage,mfq::cuda::make_contiguous_view(data(),shape,mfq::cuda::ScalarType::uint8,device));
    }
private:
    std::shared_ptr<State> state_;
};
#endif
