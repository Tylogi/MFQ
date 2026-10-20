#include "mfq/quant/nvq_rows.h"
#include "quant/packed_row_range.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq {
namespace {
template<class T> T at(const std::vector<std::uint8_t>& bytes,std::size_t offset) {
    if (offset>bytes.size() || sizeof(T)>bytes.size()-offset)
        throw std::runtime_error("truncated NVQ row metadata");
    T value; std::memcpy(&value,bytes.data()+offset,sizeof(value)); return value;
}
std::size_t packed(std::uint64_t items,int bits) {
    if (!bits) return 0;
    if (items>(std::numeric_limits<std::uint64_t>::max()-7)/bits ||
        (items*bits+7)/8>std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("NVQ row stream is too large");
    return static_cast<std::size_t>((items*bits+7)/8);
}

void unpack_group64(const std::vector<std::uint64_t>& records,std::size_t rows,int groups,int vectors,
        const std::array<NvqRows::FieldTarget,4>& fields) {
    for(int f=1;f<4;++f)if(fields[f].bytes)std::memset(fields[f].data,0,fields[f].bytes);
    std::uint64_t invalid=0;
    constexpr std::uint64_t parity_mask=(1ull<<12)|(1ull<<32)|(1ull<<52);
    for(std::size_t i=0;i<records.size();++i) {
        auto parity=records[i]^(records[i]>>4);parity^=parity>>2;parity^=parity>>1;
        invalid|=parity&parity_mask;
        fields[1].data[i/2]|=std::uint8_t((records[i]>>60)<<((i&1)*4));
    }
    if(invalid)throw std::runtime_error("invalid NVQ-JSC group64 parity bit");
    if(const int tail=vectors%3) {
        const auto padding=((1ull<<60)-1)^((1ull<<(tail*20))-1);
        for(std::size_t row=0;row<rows;++row)
            if(records[(row+1)*groups-1]&padding)
                throw std::runtime_error("NVQ-JSC group64 padding must be zero");
    }
    std::size_t vector_index=0;
    for(std::size_t row=0;row<rows;++row) {
        int group=0;
        while(group<groups) {
            // Eight complete records hold 24 vectors. Their index and sign
            // fields occupy whole bytes, allowing word stores with no OR
            // dependencies between individual signs or neighbouring indices.
            if(!(vector_index&7) && group*3+24<=vectors) {
                std::uint64_t signs[4];
                auto* indices=fields[2].data+vector_index*12/8;
                for(int pair=0;pair<4;++pair) {
                    const auto a=records[row*groups+group+pair*2];
                    const auto b=records[row*groups+group+pair*2+1];
                    const auto index_a=(a&4095)|((a>>20)&4095)<<12|((a>>40)&4095)<<24;
                    const auto index_b=(b&4095)|((b>>20)&4095)<<12|((b>>40)&4095)<<24;
                    const auto word=index_a|(index_b<<36);
                    // Six indices occupy nine bytes. A word and its tail
                    // avoid six narrow stores without writing past the field.
                    std::memcpy(indices+pair*9,&word,8);
                    indices[pair*9+8]=std::uint8_t(index_b>>28);
                    signs[pair]=((a>>12)&127)|((a>>32)&127)<<7|((a>>52)&127)<<14|
                        ((b>>12)&127)<<21|((b>>32)&127)<<28|((b>>52)&127)<<35;
                }
                const std::uint64_t low=signs[0]|signs[1]<<42;
                const std::uint64_t middle=signs[1]>>22|signs[2]<<20|signs[3]<<62;
                const std::uint64_t high=signs[3]>>2;
                auto* aux=fields[3].data+vector_index*7/8;
                std::memcpy(aux,&low,8);std::memcpy(aux+8,&middle,8);std::memcpy(aux+16,&high,5);
                vector_index+=24;group+=8;continue;
            }
            const auto record=records[row*groups+group];
            for(int local=0;local<3 && group*3+local<vectors;++local,++vector_index) {
                const auto segment=std::uint32_t((record>>(local*20))&0xfffffu);
                const auto index_byte=vector_index*12/8;
                const auto index=(segment&0xfffu)<<((vector_index&1)*4);
                fields[2].data[index_byte]|=std::uint8_t(index);
                fields[2].data[index_byte+1]|=std::uint8_t(index>>8);
                const auto sign_bit=vector_index*7,shift=sign_bit&7;
                const auto value=((segment>>12)&127u)<<shift;
                fields[3].data[sign_bit/8]|=std::uint8_t(value);
                if(shift>1)fields[3].data[sign_bit/8+1]|=std::uint8_t(value>>8);
            }
            ++group;
        }
    }
}
}

NvqRows::NvqRows(std::size_t bytes,Read read,ReadBatch batch)
    :read_(std::move(read)),read_batch_(std::move(batch)),bytes_(bytes) {
    if (!read_ || bytes<40) throw std::runtime_error("missing NVQ row header");
    prefix_.resize(40); read_(0,prefix_.data(),prefix_.size());
    const auto magic=[&](const char* value) { return std::memcmp(prefix_.data(),value,4)==0; };
    const auto profile=prefix_[4];
    std::size_t metadata=0;
    if (magic("NQ1L")) {
        if (profile!=1 && profile!=2) throw std::runtime_error("invalid NVQ1-L row profile");
        format_=1; if (profile==2) metadata=4096;
    } else if (magic("NQ1S")) {
        if (profile!=1) throw std::runtime_error("invalid NVQ1-S row profile");
        format_=8; metadata=2048;
    } else if (magic("NPQL")) {
        if (profile!=1) throw std::runtime_error("invalid NPQ0-L row profile");
        format_=7; metadata=832;
    } else if (magic("NPQS")) {
        if (profile!=2) throw std::runtime_error("invalid NPQ0-S row profile");
        format_=9; metadata=320;
    } else if (magic("NVQ1") || magic("NIQ1")) {
        const int id=profile&0x1f;
        if (profile&0x20) {
            const int formats[]={0,5,10,12,13,14,15};
            if (id<1 || id>6 || (profile&0xc0))
                throw std::runtime_error("invalid JSC row profile");
            format_=formats[id];
            if (bytes-40<64) throw std::runtime_error("truncated JSC row table header");
            std::vector<std::uint8_t> header(64); read_(40,header.data(),header.size());
            const int banks=header[1],version=header[0];
            if ((banks!=1 && banks!=2 && banks!=4) || (version!=1 && version!=2) || header[2]!=16)
                throw std::runtime_error("invalid JSC row table geometry");
            group64_=version==2;
            if ((group64_ && (format_!=14 || header[52]!=1)) || (!group64_ && header[52]!=0))
                throw std::runtime_error("invalid JSC row storage layout");
            const int entries=format_==12 ? 512 : (format_==13 || format_==15) ? 1024 : format_==14 ? 4096 : 256;
            const int vector_size=format_==10 || format_==12 || format_==15 ? 4 : 8;
            metadata=64+static_cast<std::size_t>(banks)*entries*vector_size;
        } else {
            if (id!=1 && id!=2) throw std::runtime_error("invalid legacy NVQ row profile");
            format_=id==1 ? 2 : 3;
            if ((profile&0x80) && format_!=2) throw std::runtime_error("invalid NVQ index parity profile");
            if (profile&0x40) metadata=512;
        }
    } else throw std::runtime_error("unsupported NVQ row magic");
    state_bits_=prefix_[5]; width_=at<std::int32_t>(prefix_,12);
    const auto raw_rows=at<std::uint32_t>(prefix_,36);
    if (at<std::uint16_t>(prefix_,6)!=24 || at<std::int32_t>(prefix_,8)!=0 ||
        at<std::uint32_t>(prefix_,16)!=2 || width_<=0 || !raw_rows ||
        raw_rows>std::numeric_limits<int>::max() || at<std::int64_t>(prefix_,20)!=raw_rows ||
        at<std::int64_t>(prefix_,28)!=width_ || state_bits_<1 || state_bits_>8)
        throw std::runtime_error("invalid NVQ row geometry");
    if ((format_==8 && state_bits_!=4) || (format_==7 && state_bits_!=3) ||
        (format_==9 && state_bits_!=2) ||
        ((profile&0x20) && (magic("NVQ1") || magic("NIQ1")) && state_bits_!=4))
        throw std::runtime_error("invalid NVQ row state width");
    rows_=static_cast<int>(raw_rows);
    groups_=1+(width_-1)/24; signs_=1+(width_-1)/8;
    const bool d4=format_==3 || format_==10 || format_==12 || format_==15;
    vectors_=1+(width_-1)/(d4 ? 4 : 8);
    index_bits_=format_==1 ? 11 : format_==7 ? 7 : (format_==8 || format_==12) ? 9 :
        format_==9 ? 6 : (format_==13 || format_==15) ? 10 : format_==14 ? 12 : 8;
    aux_bits_=(format_==1 || format_==8) ? 1 : (format_==7 || format_==9) ? 0 : 7;
    if (metadata>bytes-40) throw std::runtime_error("truncated NVQ row codebook");
    prefix_.resize(40+metadata);
    if (metadata) read_(40,prefix_.data()+40,metadata);
    std::size_t cursor=prefix_.size();
    const auto reserve=[&](std::size_t count) {
        if (cursor>bytes_ || count>bytes_-cursor) throw std::runtime_error("truncated NVQ row stream");
        const auto offset=cursor; cursor+=count; return offset;
    };
    anchor_=reserve(static_cast<std::size_t>(rows_)*2);
    if (group64_) states_=reserve(packed(std::uint64_t(rows_)*groups_,64));
    else {
        states_=reserve(packed(std::uint64_t(rows_)*groups_,state_bits_));
        indices_=reserve(packed(std::uint64_t(rows_)*vectors_,index_bits_));
        aux_=reserve(packed(std::uint64_t(rows_)*(aux_bits_==1 ? groups_ : signs_),aux_bits_));
    }
    if (cursor!=bytes_) throw std::runtime_error("trailing NVQ row bytes");
}

std::vector<std::uint8_t> NvqRows::slice_rows_blob(std::int64_t begin,std::int64_t end) const {
    if (begin<0 || end<=begin || end>rows_) throw std::out_of_range("NVQ contiguous row range");
    const auto rows64=end-begin; const auto rows32=static_cast<std::uint32_t>(rows64);
    std::vector<std::uint8_t> result;
    result.reserve(row_range_nbytes(begin,end));
    result.insert(result.end(),prefix_.begin(),prefix_.end());
    std::memcpy(result.data()+20,&rows64,8); std::memcpy(result.data()+36,&rows32,4);
    const auto anchor=result.size(); result.resize(anchor+static_cast<std::size_t>(rows64)*2);
    read_(anchor_+static_cast<std::size_t>(begin)*2,result.data()+anchor,static_cast<std::size_t>(rows64)*2);
    const auto field=[&](std::size_t offset,int items,int bits) {
        detail::append_packed_range(result,read_,offset,std::uint64_t(begin)*items*bits,
            std::uint64_t(rows64)*items*bits);
    };
    if (group64_) field(states_,groups_,64);
    else {
        field(states_,groups_,state_bits_); field(indices_,vectors_,index_bits_);
        field(aux_,aux_bits_==1 ? groups_ : signs_,aux_bits_);
    }
    return result;
}

std::optional<NvqRows::Fields> NvqRows::slice_rows_fields(std::int64_t begin,std::int64_t end) const {
    if(begin<0 || end<=begin || end>rows_)throw std::out_of_range("NVQ contiguous field range");
    Fields result;
    const auto rows=static_cast<std::uint64_t>(end-begin);
    result.neuron_scale.resize(static_cast<std::size_t>(rows));
    const ReadSpan anchor_span{anchor_+static_cast<std::size_t>(begin)*2,
        reinterpret_cast<std::byte*>(result.neuron_scale.data()),result.neuron_scale.size()*2};
    const auto validate_anchors=[&] {
        for(const auto anchor:result.neuron_scale)
            if((anchor&0x7c00u)==0x7c00u || (anchor&0x8000u))
                throw std::runtime_error("NVQ neuron anchors must be finite and non-negative");
    };
    if(group64_) {
        result.states.resize(packed(rows*groups_,4));
        result.indices.resize(packed(rows*vectors_,12));
        result.aux.resize(packed(rows*vectors_,7));
        const std::array<FieldTarget,4> targets{{
            {reinterpret_cast<std::uint8_t*>(result.neuron_scale.data()),result.neuron_scale.size()*2},
            {result.states.data(),result.states.size()},{result.indices.data(),result.indices.size()},
            {result.aux.data(),result.aux.size()}}};
        copy_rows_fields(begin,end,targets,&result.source_nbytes);
        return result;
    }
    const int aux_items=aux_bits_==1?groups_:signs_;
    if(read_batch_ && !(std::uint64_t(begin)*groups_*state_bits_&7) &&
            !(std::uint64_t(begin)*vectors_*index_bits_&7) &&
            !(std::uint64_t(begin)*aux_items*aux_bits_&7)) {
        std::vector<ReadSpan> spans{anchor_span};
        const auto plan=[&](auto& output,std::size_t offset,int items,int bits) {
            output.resize(packed(rows*items,bits));
            if(!output.empty())spans.push_back({offset+std::uint64_t(begin)*items*bits/8,
                reinterpret_cast<std::byte*>(output.data()),output.size()});
        };
        plan(result.states,states_,groups_,state_bits_);
        plan(result.indices,indices_,vectors_,index_bits_);
        plan(result.aux,aux_,aux_items,aux_bits_);
        read_batch_(spans);validate_anchors();
        const auto trim=[&](auto& output,std::uint64_t bits) {
            if(bits&7)output.back()&=std::uint8_t((1u<<(bits&7))-1);
        };
        trim(result.states,rows*groups_*state_bits_);
        trim(result.indices,rows*vectors_*index_bits_);
        trim(result.aux,rows*aux_items*aux_bits_);
        result.source_nbytes=result.nbytes();return result;
    }
    read_(anchor_span.offset,reinterpret_cast<std::uint8_t*>(anchor_span.destination),anchor_span.size);
    validate_anchors();
    const auto field=[&](auto& output,std::size_t offset,int items,int bits) {
        detail::append_packed_range(output,read_,offset,std::uint64_t(begin)*items*bits,rows*items*bits);
    };
    field(result.states,states_,groups_,state_bits_);
    field(result.indices,indices_,vectors_,index_bits_);
    field(result.aux,aux_,aux_bits_==1?groups_:signs_,aux_bits_);
    result.source_nbytes=result.nbytes();
    return result;
}

bool NvqRows::byte_aligned_rows(std::int64_t rows) const noexcept {
    if(rows<=0 || rows>rows_)return false;
    const auto n=std::uint64_t(rows);
    const auto auxiliary=aux_bits_==1?groups_:signs_;
    return ((n*groups_*(group64_?4:state_bits_) | n*vectors_*index_bits_ |
        n*auxiliary*aux_bits_)&7)==0;
}

bool NvqRows::copy_rows_fields(std::int64_t begin,std::int64_t end,
        const std::array<FieldTarget,4>& targets,std::size_t* source_bytes) const {
    if(begin<0 || end<=begin || end>rows_)throw std::out_of_range("NVQ direct field range");
    const auto rows=std::uint64_t(end-begin);
    if(group64_) {
        const std::array<std::size_t,4> sizes={std::size_t(rows)*2,packed(rows*groups_,4),
            packed(rows*vectors_,12),packed(rows*vectors_,7)};
        for(int f=0;f<4;++f)if(targets[f].bytes!=sizes[f] || (sizes[f] && !targets[f].data))
            throw std::invalid_argument("NVQ group64 target size differs");
        std::vector<std::uint64_t> records(std::size_t(rows)*groups_);
        const std::vector<ReadSpan> spans{
            {anchor_+std::size_t(begin)*2,reinterpret_cast<std::byte*>(targets[0].data),sizes[0]},
            {states_+std::size_t(begin)*groups_*8,reinterpret_cast<std::byte*>(records.data()),records.size()*8}};
        if(read_batch_)read_batch_(spans);
        else for(const auto& span:spans)read_(span.offset,reinterpret_cast<std::uint8_t*>(span.destination),span.size);
        for(std::size_t row=0;row<std::size_t(rows);++row) {
            std::uint16_t anchor;std::memcpy(&anchor,targets[0].data+row*2,2);
            if((anchor&0x7c00u)==0x7c00u || (anchor&0x8000u))
                throw std::runtime_error("NVQ neuron anchors must be finite and non-negative");
        }
        unpack_group64(records,std::size_t(rows),groups_,vectors_,targets);
        if(source_bytes)*source_bytes=sizes[0]+records.size()*8;
        return true;
    }
    const int aux_items=aux_bits_==1?groups_:signs_;
    const std::array<std::uint64_t,4> first={std::uint64_t(begin)*16,
        std::uint64_t(begin)*groups_*state_bits_,std::uint64_t(begin)*vectors_*index_bits_,
        std::uint64_t(begin)*aux_items*aux_bits_};
    const std::array<std::uint64_t,4> counts={rows*16,rows*groups_*state_bits_,
        rows*vectors_*index_bits_,rows*aux_items*aux_bits_};
    const std::array<std::size_t,4> offsets={anchor_,states_,indices_,aux_};
    for(int f=0;f<4;++f)if((first[f]|counts[f])&7)return false;
    std::vector<ReadSpan> spans;
    for(int f=0;f<4;++f) {
        if(targets[f].bytes!=counts[f]/8 || (targets[f].bytes && !targets[f].data))
            throw std::invalid_argument("NVQ direct field target size differs");
        if(targets[f].bytes)spans.push_back({offsets[f]+first[f]/8,
            reinterpret_cast<std::byte*>(targets[f].data),targets[f].bytes});
    }
    if(read_batch_)read_batch_(spans);
    else for(const auto& span:spans)read_(span.offset,reinterpret_cast<std::uint8_t*>(span.destination),span.size);
    for(std::size_t row=0;row<std::size_t(rows);++row) {
        std::uint16_t anchor;std::memcpy(&anchor,targets[0].data+row*2,2);
        if((anchor&0x7c00u)==0x7c00u || (anchor&0x8000u))
            throw std::runtime_error("NVQ neuron anchors must be finite and non-negative");
    }
    if(source_bytes)*source_bytes=targets[0].bytes+targets[1].bytes+targets[2].bytes+targets[3].bytes;
    return true;
}

std::size_t NvqRows::row_range_nbytes(std::int64_t begin,std::int64_t end) const {
    if (begin<0 || end<=begin || end>rows_) throw std::out_of_range("NVQ contiguous row range");
    const auto rows=static_cast<std::uint64_t>(end-begin);
    std::size_t bytes=prefix_.size()+static_cast<std::size_t>(rows)*2;
    if (group64_) bytes+=packed(rows*groups_,64);
    else bytes+=packed(rows*groups_,state_bits_)+packed(rows*vectors_,index_bits_)+
        packed(rows*(aux_bits_==1 ? groups_ : signs_),aux_bits_);
    return bytes;
}
} // namespace mfq
