#include "mfq/nvq_rows.h"
#include "mfq/packed_row_range.h"

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
}

NvqRows::NvqRows(std::size_t bytes,Read read):read_(std::move(read)),bytes_(bytes) {
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
