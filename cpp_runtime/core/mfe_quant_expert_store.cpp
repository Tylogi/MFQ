#include "mfq/mfe_quant_expert_store.h"
#include "mfq_format_compat.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq {
namespace {
template<class T> T value(const std::vector<std::uint8_t>& data,std::size_t offset) {
    if (offset>data.size() || sizeof(T)>data.size()-offset)
        throw std::runtime_error("truncated mixed expert metadata");
    T result; std::memcpy(&result,data.data()+offset,sizeof(result)); return result;
}
}

MfeQuantExpertStore::MfeQuantExpertStore(std::size_t bytes,Read reader):read_(std::move(reader)),bytes_(bytes) {
    if (!read_ || bytes_<20)
        throw std::runtime_error("missing mixed expert range source");
    const auto read=[&](std::uint64_t offset,std::size_t bytes) {
        if (offset>bytes_ || bytes>bytes_-offset)
            throw std::runtime_error("truncated mixed expert range source");
        std::vector<std::uint8_t> result(bytes);
        read_(static_cast<std::size_t>(offset),result.data(),result.size()); return result;
    };
    const auto header=read(0,20);
    if (std::memcmp(header.data(),"MFE1",4) && std::memcmp(header.data(),"NIM2",4))
        throw std::runtime_error("mixed expert range source requires canonical MFE1");
    const auto experts=value<std::uint32_t>(header,4),output=value<std::uint32_t>(header,8),
        width=value<std::uint32_t>(header,12),pools=value<std::uint32_t>(header,16);
    if (!experts || !output || !width || !pools || pools>experts ||
        experts>std::numeric_limits<int>::max() || output>std::numeric_limits<int>::max() ||
        width>std::numeric_limits<int>::max()) throw std::runtime_error("invalid mixed expert geometry");
    experts_=static_cast<int>(experts); output_=static_cast<int>(output); width_=static_cast<int>(width);
    owners_.resize(experts); pools_.reserve(pools);
    std::uint64_t cursor=20;
    for (std::uint32_t index=0; index<pools; ++index) {
        const auto pool_header=read(cursor,24); cursor+=24;
        const auto count=value<std::uint32_t>(pool_header,0),dtype_bytes=value<std::uint32_t>(pool_header,4);
        const auto bytes=value<std::uint64_t>(pool_header,8),runtime_bytes=value<std::uint64_t>(pool_header,16);
        if (!count || count>experts || count>std::numeric_limits<int>::max()/output ||
            !dtype_bytes || dtype_bytes>32)
            throw std::runtime_error("invalid mixed expert pool metadata");
        if (runtime_bytes) throw MfeQuantRangeUnsupported("mixed expert embedded runtime fields require the full codec");
        const auto metadata=read(cursor,std::size_t(count)*4+dtype_bytes);
        cursor+=metadata.size();
        if (bytes>std::numeric_limits<std::size_t>::max() || cursor>bytes_ || bytes>bytes_-cursor)
            throw std::runtime_error("truncated mixed expert pool payload");
        Pool pool;
        pool.dtype=std::string(mfq::canonical_format_dtype(std::string(
            reinterpret_cast<const char*>(metadata.data()+std::size_t(count)*4),dtype_bytes)));
        pool.ids.resize(count);
        for (std::uint32_t local=0; local<count; ++local) {
            const auto expert=value<std::int32_t>(metadata,std::size_t(local)*4);
            if (expert<0 || expert>=experts_ || owners_[expert].pool>=0)
                throw std::runtime_error("invalid mixed expert ownership");
            pool.ids[local]=expert;
            owners_[expert]={static_cast<int>(index),static_cast<int>(local)};
        }
        // Capture the immutable range callback rather than the ModelSource
        // owner by reference; callers supply its owning tensor_reader handle.
        const auto body=[reader=read_,offset=static_cast<std::size_t>(cursor),bytes](std::size_t position,
                std::uint8_t* output,std::size_t size) {
            if (position>bytes || size>bytes-position) throw std::runtime_error("mixed expert row read exceeds pool");
            reader(offset+position,output,size);
        };
        int rows=0,columns=0;
        if (pool.dtype=="NINT") {
            pool.nint=std::make_shared<mfq::NintRows>(static_cast<std::size_t>(bytes),body);
            rows=pool.nint->rows(); columns=pool.nint->width();
        } else if (pool.dtype=="NVQ" || pool.dtype=="NPQ") {
            pool.nvq=std::make_shared<mfq::NvqRows>(static_cast<std::size_t>(bytes),body);
            rows=pool.nvq->rows(); columns=pool.nvq->width();
        } else throw MfeQuantRangeUnsupported("unsupported mixed expert range dtype: "+pool.dtype);
        if (rows!=static_cast<int>(count*output) || columns!=width_)
            throw std::runtime_error("mixed expert pool shape mismatch");
        pools_.push_back(std::move(pool)); cursor+=bytes;
    }
    if (cursor!=bytes_) throw std::runtime_error("mixed expert range tail mismatch");
    for (const auto& owner: owners_)
        if (owner.pool<0) throw std::runtime_error("mixed expert coverage is incomplete");
}

std::size_t MfeQuantExpertStore::index_nbytes() const noexcept {
    std::size_t bytes=owners_.size()*sizeof(Expert);
    for (const auto& pool: pools_)
        bytes+=pool.ids.size()*sizeof(std::int32_t)+pool.dtype.size()+
            (pool.nint ? pool.nint->index_nbytes() : pool.nvq->index_nbytes());
    return bytes;
}

const std::string& MfeQuantExpertStore::expert_dtype(int expert) const {
    if (expert<0 || expert>=experts_) throw std::out_of_range("mixed expert ID");
    return pools_[owners_[expert].pool].dtype;
}

MfeQuantExpert MfeQuantExpertStore::read_expert(int expert) const {
    if (expert<0 || expert>=experts_) throw std::out_of_range("mixed expert ID");
    const auto& owner=owners_[expert]; const auto& pool=pools_[owner.pool];
    const auto begin=std::int64_t(owner.local)*output_,end=begin+output_;
    return {pool.dtype,pool.nint ? pool.nint->slice_rows_blob(begin,end) : pool.nvq->slice_rows_blob(begin,end)};
}

std::size_t MfeQuantExpertStore::expert_payload_nbytes(int expert) const {
    if (expert<0 || expert>=experts_) throw std::out_of_range("mixed expert ID");
    const auto& owner=owners_[expert]; const auto& pool=pools_[owner.pool];
    const auto begin=std::int64_t(owner.local)*output_,end=begin+output_;
    return pool.nint ? pool.nint->row_range_nbytes(begin,end) : pool.nvq->row_range_nbytes(begin,end);
}

int MfeQuantExpertStore::expert_pool(int expert) const {
    if (expert<0 || expert>=experts_) throw std::out_of_range("mixed expert ID");
    return owners_[expert].pool;
}

std::uint64_t MfeQuantExpertStore::expert_values_bits(int expert) const {
    const auto& owner=owners_.at(static_cast<std::size_t>(expert));
    const auto& pool=pools_[owner.pool];
    if (!pool.nint) return 0;
    const auto begin=std::int64_t(owner.local)*output_;
    return pool.nint->row_values_bits(begin,begin+output_);
}
} // namespace mfq
