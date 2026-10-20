#include "mfq/quant/mfe_expert_store.h"
#include "mfq/quant/packed_nint_expert_store.h"
#include "compat/mfq_format_compat.h"
#include "mfq/host_parallel.h"

#include <algorithm>
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
template<int Bits> void copy_nint_metadata(const std::uint8_t* source,std::size_t bytes,
        std::uint64_t first,std::size_t count,std::uint8_t* target) {
    constexpr std::uint32_t mask=(1u<<Bits)-1;
    const auto one=[&] {
        const auto byte=std::size_t(first/8);std::uint32_t word=source[byte];
        if(byte+1<bytes)word|=std::uint32_t(source[byte+1])<<8;
        *target++=std::uint8_t((word>>(first&7))&mask);first+=Bits;--count;
    };
    while(count && (first&7))one();
    if constexpr(Bits==8) {
        if(count)std::memcpy(target,source+std::size_t(first/8),count);
        return;
    }
    while(count>=8) {
        std::uint64_t word=0;std::memcpy(&word,source+std::size_t(first/8),Bits);
        for(int i=0;i<8;++i)target[i]=std::uint8_t((word>>(i*Bits))&mask);
        target+=8;first+=8*Bits;count-=8;
    }
    while(count)one();
}
void copy_nint_metadata(const std::uint8_t* source,std::size_t bytes,std::uint64_t first,
        std::size_t count,int bits,std::uint8_t* target) {
    switch(bits) {
        case 1:copy_nint_metadata<1>(source,bytes,first,count,target);break;
        case 2:copy_nint_metadata<2>(source,bytes,first,count,target);break;
        case 3:copy_nint_metadata<3>(source,bytes,first,count,target);break;
        case 4:copy_nint_metadata<4>(source,bytes,first,count,target);break;
        case 5:copy_nint_metadata<5>(source,bytes,first,count,target);break;
        case 6:copy_nint_metadata<6>(source,bytes,first,count,target);break;
        case 7:copy_nint_metadata<7>(source,bytes,first,count,target);break;
        case 8:copy_nint_metadata<8>(source,bytes,first,count,target);break;
        default:throw std::runtime_error("invalid NINT metadata field width");
    }
}
}

void copy_nint_runtime_fields(const std::vector<std::uint8_t>& blob,
        const std::array<QuantFieldTarget,7>& targets) {
    if(blob.size()<42)throw std::runtime_error("truncated NINT runtime field source");
    const bool adaptive=bool(blob[0]&128);
    const auto nominal=blob[0]&127,sub_bits=int(blob[1]);
    const auto gs=value<std::int32_t>(blob,2),width=value<std::int32_t>(blob,10);
    const auto rows=value<std::uint32_t>(blob,34),groups=value<std::uint32_t>(blob,38);
    if(nominal<1 || nominal>8 || sub_bits<1 || sub_bits>8 || gs<=0 || width<=0 || !rows || !groups ||
        value<std::int32_t>(blob,6)!=0 || value<std::uint32_t>(blob,14)!=2 ||
        value<std::int64_t>(blob,18)!=rows || value<std::int64_t>(blob,26)!=width ||
        (std::uint64_t(width)+gs-1)/gs!=groups)
        throw std::runtime_error("invalid NINT runtime field geometry");
    const std::uint64_t values=std::uint64_t(groups)*gs;
    std::size_t offset=42;
    const auto bytes_for=[](std::uint64_t count,int bits) {
        if(count>(std::numeric_limits<std::uint64_t>::max()-7)/bits ||
            (count*bits+7)/8>std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("NINT runtime field size overflow");
        return std::size_t((count*bits+7)/8);
    };
    const auto take=[&](std::size_t bytes) {
        if(offset>blob.size() || bytes>blob.size()-offset)throw std::runtime_error("truncated NINT runtime field stream");
        const auto* start=blob.data()+offset;offset+=bytes;return start;
    };
    const auto* neuron_scale=take(std::size_t(rows)*2);
    const auto* neuron_min=take(std::size_t(rows)*2);
    const auto unpack=[](const std::uint8_t* stream,std::size_t bytes,std::uint64_t bit,int bits) {
        const auto byte=std::size_t(bit/8);
        std::uint32_t word=stream[byte];
        if(byte+1<bytes)word|=std::uint32_t(stream[byte+1])<<8;
        return std::uint8_t((word>>(bit&7))&((1u<<bits)-1));
    };
    std::vector<std::uint8_t> ks(rows,0),qs(rows,0);
    std::array<std::uint32_t,4> kcounts{};
    std::array<std::uint32_t,8> qcounts{};
    if(adaptive) {
        const auto bytes=bytes_for(rows,2);const auto* stream=take(bytes);
        for(std::size_t row=0;row<rows;++row)ks[row]=unpack(stream,bytes,row*2,2);
    }
    for(const auto k:ks)++kcounts[k];
    struct Stream {const std::uint8_t* data=nullptr;std::size_t bytes=0;};
    std::array<Stream,4> scale{},minimum{};
    for(int k=0;k<4;++k)if(kcounts[k]) {
        const auto bits=adaptive?sub_bits-1+k:sub_bits;
        if(bits<1 || bits>8)throw std::runtime_error("invalid NINT runtime subgroup width");
        const auto bytes=bytes_for(std::uint64_t(kcounts[k])*groups,bits);
        scale[k]={take(bytes),bytes};minimum[k]={take(bytes),bytes};
    }
    if(adaptive) {
        const auto bytes=bytes_for(rows,3);const auto* stream=take(bytes);
        for(std::size_t row=0;row<rows;++row)qs[row]=unpack(stream,bytes,row*3,3);
    }
    for(const auto q:qs)++qcounts[q];
    std::array<Stream,8> qstreams{};
    std::uint64_t qbits=0;
    for(int q=0;q<8;++q)if(qcounts[q]) {
        const auto bits=adaptive?q+1:nominal;
        if(values>std::numeric_limits<std::uint64_t>::max()/qcounts[q])
            throw std::overflow_error("NINT runtime value count overflow");
        const auto count=std::uint64_t(qcounts[q])*values;
        const auto bytes=bytes_for(count,bits);
        qstreams[q]={take(bytes),bytes};
        if(count*bits>std::uint64_t(INT64_MAX)-qbits)throw std::overflow_error("NINT runtime bit offsets overflow");
        qbits+=count*bits;
    }
    if(offset!=blob.size())throw std::runtime_error("trailing NINT runtime field bytes");
    const std::array<std::size_t,7> sizes={bytes_for(qbits,1)+8,rows,std::size_t(rows)*8,
        std::size_t(rows)*groups,std::size_t(rows)*groups,std::size_t(rows)*2,std::size_t(rows)*2};
    for(int f=0;f<7;++f)
        if((f==0?targets[f].bytes<sizes[f]:targets[f].bytes!=sizes[f]) ||
            (targets[f].bytes && !targets[f].data))throw std::invalid_argument("NINT runtime field target size differs");
    for(const auto* anchors:{neuron_scale,neuron_min})for(std::size_t row=0;row<rows;++row) {
        std::uint16_t anchor;std::memcpy(&anchor,anchors+row*2,2);
        if((anchor&0x7c00u)==0x7c00u)throw std::runtime_error("NINT neuron metadata must be finite");
    }
    const auto payload_bytes=bytes_for(qbits,1);
    std::memset(targets[0].data+payload_bytes,0,targets[0].bytes-payload_bytes);
    std::memcpy(targets[5].data,neuron_scale,sizes[5]);std::memcpy(targets[6].data,neuron_min,sizes[6]);
    std::array<std::uint32_t,4> knext{};
    std::array<std::uint32_t,8> qnext{};
    std::uint64_t destination=0;
    for(std::size_t row=0;row<rows;++row) {
        const auto q=qs[row],k=ks[row];const auto qb=adaptive?q+1:nominal,kb=adaptive?sub_bits-1+k:sub_bits;
        targets[1].data[row]=std::uint8_t(qb);
        const auto bit_offset=std::int64_t(destination);std::memcpy(targets[2].data+row*8,&bit_offset,8);
        auto first=std::uint64_t(qnext[q]++)*values*qb,length=values*qb;
        const auto* input=qstreams[q].data;
        if(!((first|destination|length)&7)) {
            std::memcpy(targets[0].data+std::size_t(destination/8),input+std::size_t(first/8),std::size_t(length/8));
            destination+=length;
        } else while(length) {
            const auto source_shift=unsigned(first&7),destination_shift=unsigned(destination&7);
            const auto bits=unsigned(std::min<std::uint64_t>(length,std::min(8-source_shift,8-destination_shift)));
            const auto v=(input[std::size_t(first/8)]>>source_shift)&((1u<<bits)-1);
            if(!destination_shift)targets[0].data[std::size_t(destination/8)]=0;
            targets[0].data[std::size_t(destination/8)]|=std::uint8_t(v<<destination_shift);
            first+=bits;destination+=bits;length-=bits;
        }
        const auto begin=std::uint64_t(knext[k]++)*groups*kb;
        for(int f=3;f<5;++f) {
            const auto& stream=f==3?scale[k]:minimum[k];
            copy_nint_metadata(stream.data,stream.bytes,begin,groups,kb,targets[f].data+row*groups);
        }
    }
}

std::shared_ptr<MfeQuantExpertStore> make_packed_nint_expert_store(
        std::size_t bytes, MfeQuantExpertStore::Read read,NintRows::ReadBatch batch) {
    if (!read || bytes < 50) throw std::runtime_error("truncated packed NINT expert header");
    std::vector<std::uint8_t> original(50);
    read(0, original.data(), original.size());
    const auto experts = value<std::int64_t>(original, 18);
    const auto output = value<std::int64_t>(original, 26);
    const auto width = value<std::int64_t>(original, 34);
    if (value<std::uint32_t>(original, 14) != 3 || experts <= 0 || output <= 0 || width <= 0 ||
            experts > std::numeric_limits<int>::max() / output ||
            width > std::numeric_limits<int>::max() ||
            experts * output > (bytes - 50) / 4 ||
            value<std::int32_t>(original, 6) != 0 ||
            value<std::int32_t>(original, 10) != width ||
            value<std::uint32_t>(original, 42) != experts * output)
        throw std::runtime_error("invalid packed NINT expert geometry");
    auto prefix = std::make_shared<std::vector<std::uint8_t>>();
    auto append = [&](auto scalar) {
        const auto offset = prefix->size();
        prefix->resize(offset + sizeof(scalar));
        std::memcpy(prefix->data() + offset, &scalar, sizeof(scalar));
    };
    prefix->insert(prefix->end(), {'M', 'F', 'E', '1'});
    for (auto dimension : {experts, output, width, std::int64_t(1)})
        append(static_cast<std::uint32_t>(dimension));
    append(static_cast<std::uint32_t>(experts));
    append(std::uint32_t(4));
    append(static_cast<std::uint64_t>(bytes - 8));
    append(std::uint64_t(0));
    for (std::int32_t expert = 0; expert < experts; ++expert) append(expert);
    prefix->insert(prefix->end(), {'N', 'I', 'N', 'T'});
    prefix->insert(prefix->end(), original.begin(), original.begin() + 14);
    append(std::uint32_t(2));
    append(static_cast<std::int64_t>(experts * output));
    append(width);
    prefix->insert(prefix->end(), original.begin() + 42, original.end());
    if (bytes - 50 > std::numeric_limits<std::size_t>::max() - prefix->size())
        throw std::overflow_error("packed NINT expert source size overflow");
    const auto size = prefix->size() + bytes - 50;
    NintRows::ReadBatch virtual_batch;
    if(batch)virtual_batch=[prefix,batch=std::move(batch),size](const std::vector<ReadSpan>& spans) {
        for(const auto& span:spans) {
            if(span.offset>size || span.size>size-span.offset)
                throw std::runtime_error("packed NINT expert batch exceeds source");
            if(span.size && !span.destination)throw std::invalid_argument("null packed NINT batch target");
        }
        std::vector<ReadSpan> physical;physical.reserve(spans.size());
        for(auto span:spans) {
            if(!span.size)continue;
            if(span.offset<prefix->size()) {
                const auto count=std::min(span.size,prefix->size()-std::size_t(span.offset));
                if(count)std::memcpy(span.destination,prefix->data()+span.offset,count);
                span.offset+=count;span.destination+=count;span.size-=count;
            }
            if(span.size){span.offset=span.offset-prefix->size()+50;physical.push_back(span);}
        }
        if(!physical.empty())batch(physical);
    };
    return std::make_shared<MfeQuantExpertStore>(size,
        [prefix, read = std::move(read), size](std::size_t offset, std::uint8_t* out, std::size_t count) {
            if (offset > size || count > size - offset)
                throw std::runtime_error("packed NINT expert read exceeds source");
            if (offset < prefix->size()) {
                const auto copied = std::min(count, prefix->size() - offset);
                std::memcpy(out, prefix->data() + offset, copied);
                offset += copied; out += copied; count -= copied;
            }
            if (count) read(offset - prefix->size() + 50, out, count);
        },std::move(virtual_batch));
}

MfeQuantExpertStore::MfeQuantExpertStore(std::size_t bytes,Read reader,NintRows::ReadBatch batch)
    :read_(std::move(reader)),read_batch_(std::move(batch)),bytes_(bytes) {
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
        pool.offset=static_cast<std::size_t>(cursor);
        pool.bytes=static_cast<std::size_t>(bytes);
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
        NintRows::ReadBatch body_batch;
        if(read_batch_)body_batch=[reader=read_batch_,offset=cursor,bytes](const std::vector<ReadSpan>& spans) {
            auto absolute=spans;
            for(auto& span:absolute) {
                if(span.offset>bytes || span.size>bytes-span.offset)
                    throw std::runtime_error("mixed expert batch read exceeds pool");
                span.offset+=offset;
            }
            reader(absolute);
        };
        int rows=0,columns=0;
        if (pool.dtype=="NINT") {
            pool.nint=std::make_shared<mfq::NintRows>(static_cast<std::size_t>(bytes),body,false,body_batch);
            rows=pool.nint->rows(); columns=pool.nint->width();
        } else if (pool.dtype=="NVQ" || pool.dtype=="NPQ") {
            pool.nvq=std::make_shared<mfq::NvqRows>(static_cast<std::size_t>(bytes),body,body_batch);
            rows=pool.nvq->rows(); columns=pool.nvq->width();
        } else if (pool.dtype=="MXFP4") {
            const auto mx=read(cursor,56);
            const std::uint64_t expected_rows=std::uint64_t(count)*output;
            if(width%32 || bytes<56 || std::memcmp(mx.data(),"MXT1",4) || mx[4]!=1 || mx[5]!=4 ||
                value<std::uint16_t>(mx,6)!=0 || value<std::uint64_t>(mx,8)!=expected_rows ||
                value<std::uint64_t>(mx,16)!=width || value<std::uint64_t>(mx,24)!=expected_rows ||
                value<std::uint64_t>(mx,32)!=width/2 || value<std::uint64_t>(mx,40)!=expected_rows ||
                value<std::uint64_t>(mx,48)!=width/32 ||
                expected_rows>(bytes-56)/(width/2+width/32) ||
                bytes-56!=expected_rows*(width/2+width/32))
                throw std::runtime_error("invalid mixed MXFP4 row layout");
            pool.mx_values=pool.offset+56;
            pool.mx_scales=pool.mx_values+std::size_t(expected_rows)*(width/2);
            rows=static_cast<int>(expected_rows);columns=width_;
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
            (pool.nint ? pool.nint->index_nbytes() : pool.nvq ? pool.nvq->index_nbytes() : 2*sizeof(std::size_t));
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
    if(pool.dtype=="MXFP4") {
        MfeQuantExpert result{pool.dtype,{'M','X','T','1',1,4,0,0}};
        for(const auto dimension:{std::uint64_t(output_),std::uint64_t(width_),std::uint64_t(output_),
                std::uint64_t(width_/2),std::uint64_t(output_),std::uint64_t(width_/32)}) {
            const auto offset=result.payload.size();result.payload.resize(offset+8);
            std::memcpy(result.payload.data()+offset,&dimension,8);
        }
        const auto values=std::size_t(output_)*(width_/2),scales=std::size_t(output_)*(width_/32);
        result.payload.resize(56+values+scales);
        copy_mxfp4_fields(expert,{{{result.payload.data()+56,values},
            {result.payload.data()+56+values,scales}}});
        return result;
    }
    return {pool.dtype,pool.nint ? pool.nint->slice_rows_blob(begin,end) : pool.nvq->slice_rows_blob(begin,end)};
}

bool MfeQuantExpertStore::copy_mxfp4_fields(int expert,const std::array<QuantFieldTarget,2>& targets,
        std::size_t* source_bytes) const {
    if(expert<0 || expert>=experts_)throw std::out_of_range("mixed MXFP4 expert ID");
    const auto& owner=owners_[expert];const auto& pool=pools_[owner.pool];
    if(pool.dtype!="MXFP4")return false;
    const std::array<std::size_t,2> sizes{std::size_t(output_)*(width_/2),std::size_t(output_)*(width_/32)};
    for(std::size_t field=0;field<2;++field)
        if(targets[field].bytes!=sizes[field] || !targets[field].data)
            throw std::invalid_argument("MXFP4 expert field target size differs");
    const std::array<std::size_t,2> starts{pool.mx_values,pool.mx_scales};
    std::vector<ReadSpan> spans;
    for(std::size_t field=0;field<2;++field)
        spans.push_back({starts[field]+std::size_t(owner.local)*sizes[field],
            reinterpret_cast<std::byte*>(targets[field].data),sizes[field]});
    if(read_batch_)read_batch_(spans);
    else for(const auto& span:spans)read_(span.offset,reinterpret_cast<std::uint8_t*>(span.destination),span.size);
    if(std::find(targets[1].data,targets[1].data+sizes[1],255)!=targets[1].data+sizes[1])
        throw std::runtime_error("MXFP4 expert contains a NaN scale");
    if(source_bytes)*source_bytes=sizes[0]+sizes[1];
    return true;
}

std::optional<NvqRows::Fields> MfeQuantExpertStore::read_nvq_fields(int expert) const {
    if(expert<0 || expert>=experts_)throw std::out_of_range("mixed expert field ID");
    const auto& owner=owners_[expert];const auto& pool=pools_[owner.pool];
    if(!pool.nvq)return std::nullopt;
    const auto begin=std::int64_t(owner.local)*output_;
    return pool.nvq->slice_rows_fields(begin,begin+output_);
}

bool MfeQuantExpertStore::copy_nvq_fields(int expert,const std::array<NvqRows::FieldTarget,4>& targets,
        std::size_t* source_bytes) const {
    if(expert<0 || expert>=experts_)throw std::out_of_range("mixed expert direct field ID");
    const auto& owner=owners_[expert];const auto& pool=pools_[owner.pool];
    if(!pool.nvq)return false;
    const auto begin=std::int64_t(owner.local)*output_;
    return pool.nvq->copy_rows_fields(begin,begin+output_,targets,source_bytes);
}

bool MfeQuantExpertStore::copy_nvq_fields(const std::vector<int>& experts,
        const std::array<NvqRows::FieldTarget,4>& targets,std::size_t* source_bytes) const {
    if(experts.empty())throw std::invalid_argument("missing NVQ expert group");
    for(int expert:experts)if(expert<0 || expert>=experts_)throw std::out_of_range("NVQ expert group ID");
    const auto& first=owners_[experts.front()];const auto& pool=pools_[first.pool];
    if(!pool.nvq || !pool.nvq->byte_aligned_rows(output_))return false;
    for(std::size_t i=0;i<experts.size();++i) {
        const auto& owner=owners_[experts[i]];
        if(owner.pool!=first.pool || std::size_t(owner.local)!=std::size_t(first.local)+i)return false;
    }
    const auto begin=std::int64_t(first.local)*output_;
    return pool.nvq->copy_rows_fields(begin,begin+std::int64_t(experts.size())*output_,targets,source_bytes);
}

bool MfeQuantExpertStore::copy_nint_fields(int expert,const std::array<QuantFieldTarget,7>& targets,
        std::size_t* source_bytes) const {
    if(expert<0 || expert>=experts_)throw std::out_of_range("mixed expert direct field ID");
    const auto& owner=owners_[expert];const auto& pool=pools_[owner.pool];
    if(!pool.nint)return false;
    const auto begin=std::int64_t(owner.local)*output_;
    const auto blob=pool.nint->slice_rows_blob(begin,begin+output_);
    copy_nint_runtime_fields(blob,targets);
    if(source_bytes)*source_bytes=blob.size();
    return true;
}

void MfeQuantExpertStore::visit_pool_experts(std::size_t index,
        const std::function<void(int, MfeQuantExpert)>& visitor,
        HostParallelPool* workers,int threads) const {
    if (!visitor) throw std::invalid_argument("missing expert preload visitor");
    const auto& pool=pools_.at(index);
    if(pool.dtype=="MXFP4") {
        const auto visit=[&](int64_t begin,int64_t end) {
            for(auto local=begin;local<end;++local)visitor(pool.ids[local],read_expert(pool.ids[local]));
        };
        if(workers)workers->run(0,pool.ids.size(),1,threads,visit);
        else visit(0,pool.ids.size());
        return;
    }
    std::vector<std::uint8_t> payload(pool.bytes);
    read_(pool.offset,payload.data(),payload.size());
    const Read memory=[&payload](std::size_t offset,std::uint8_t* destination,std::size_t bytes) {
        if (offset>payload.size() || bytes>payload.size()-offset)
            throw std::out_of_range("expert preload pool bounds");
        if (bytes) std::memcpy(destination,payload.data()+offset,bytes);
    };
    if (pool.nint) {
        NintRows rows(payload.size(),memory);
        const auto visit=[&](int64_t begin,int64_t end) {
            for(auto local=begin;local<end;++local)
                visitor(pool.ids[local],{pool.dtype,rows.slice_rows_blob(local*output_,(local+1)*output_)});
        };
        if(workers)workers->run(0,pool.ids.size(),1,threads,visit);
        else visit(0,pool.ids.size());
    } else {
        NvqRows rows(payload.size(),memory);
        const auto visit=[&](int64_t begin,int64_t end) {
            for(auto local=begin;local<end;++local)
                visitor(pool.ids[local],{pool.dtype,rows.slice_rows_blob(local*output_,(local+1)*output_)});
        };
        if(workers)workers->run(0,pool.ids.size(),1,threads,visit);
        else visit(0,pool.ids.size());
    }
}

std::size_t MfeQuantExpertStore::expert_payload_nbytes(int expert) const {
    if (expert<0 || expert>=experts_) throw std::out_of_range("mixed expert ID");
    const auto& owner=owners_[expert]; const auto& pool=pools_[owner.pool];
    const auto begin=std::int64_t(owner.local)*output_,end=begin+output_;
    if(pool.dtype=="MXFP4")return 56+std::size_t(output_)*(width_/2+width_/32);
    return pool.nint ? pool.nint->row_range_nbytes(begin,end) : pool.nvq->row_range_nbytes(begin,end);
}

int MfeQuantExpertStore::expert_pool(int expert) const {
    if (expert<0 || expert>=experts_) throw std::out_of_range("mixed expert ID");
    return owners_[expert].pool;
}

int MfeQuantExpertStore::expert_local(int expert) const {
    if(expert<0 || expert>=experts_)throw std::out_of_range("mixed expert ID");
    return owners_[expert].local;
}

std::uint64_t MfeQuantExpertStore::expert_values_bits(int expert) const {
    const auto& owner=owners_.at(static_cast<std::size_t>(expert));
    const auto& pool=pools_[owner.pool];
    if (!pool.nint) return 0;
    const auto begin=std::int64_t(owner.local)*output_;
    return pool.nint->row_values_bits(begin,begin+output_);
}
} // namespace mfq
