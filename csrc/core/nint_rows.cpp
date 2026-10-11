#include "mfq/nint_rows.h"

#include "mfq/nint_blob.h"
#include "mfq/host_parallel.h"
#include "mfq/packed_row_range.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mfq {
namespace {

constexpr std::size_t kRankStride = 256;

std::size_t packed_size(std::uint64_t count, int bits) {
    if (count > (std::numeric_limits<std::uint64_t>::max() - 7) / bits) {
        throw std::overflow_error("NINT row stream size overflow");
    }
    return static_cast<std::size_t>((count * bits + 7) / 8);
}

class Cursor {
public:
    Cursor(std::size_t size, const NintRows::Read& read) : size_(size) {
        if (size<sizeof(header_)) throw std::runtime_error("truncated mapped NINT row header");
        read(0,header_,sizeof(header_));
    }
    std::size_t skip(std::size_t count) {
        if (offset_ > size_ || count > size_ - offset_) {
            throw std::runtime_error("truncated mapped NINT row tensor");
        }
        auto start = offset_;
        offset_ += count;
        return start;
    }
    template <typename T> T scalar() {
        const auto start = skip(sizeof(T));
        T value;
        if (start>sizeof(header_) || sizeof(T)>sizeof(header_)-start)
            throw std::runtime_error("invalid mapped NINT row header offset");
        std::memcpy(&value,header_+start,sizeof(T));
        return value;
    }
    bool finished() const noexcept { return offset_ == size_; }
private:
    std::size_t size_;
    std::uint8_t header_[42];
    std::size_t offset_ = 0;
};

std::uint32_t selector(
    const std::uint8_t* stream, std::size_t row, int bits) {
    const auto bit = row * static_cast<std::size_t>(bits);
    const auto byte = bit / 8;
    std::uint32_t value = stream[byte];
    if ((bit & 7) + bits > 8) value |= std::uint32_t(stream[byte + 1]) << 8;
    return (value >> (bit & 7)) & ((1u << bits) - 1);
}

template <std::size_t N>
std::array<std::uint32_t, N> build_ranks(
    const std::uint8_t* selectors, int rows, int bits,
    std::vector<std::array<std::uint32_t, N>>& checkpoints) {
    std::array<std::uint32_t, N> counts{};
    checkpoints.reserve((static_cast<std::size_t>(rows) + kRankStride - 1) / kRankStride);
    for (int row = 0; row < rows; ++row) {
        if (row % kRankStride == 0) checkpoints.push_back(counts);
        ++counts[selector(selectors, row, bits)];
    }
    return counts;
}

template <std::size_t N>
std::uint32_t cohort_rank(
    const std::uint8_t* selectors, std::size_t row, int bits,
    std::uint32_t cohort,
    const std::vector<std::array<std::uint32_t, N>>& checkpoints) {
    auto rank = checkpoints[row / kRankStride][cohort];
    for (auto index = row - row % kRankStride; index < row; ++index) {
        rank += selector(selectors, index, bits) == cohort;
    }
    return rank;
}

} // namespace

NintRows::NintRows(const std::uint8_t* data, std::size_t size)
    : read_([data](std::size_t offset, std::uint8_t* out, std::size_t count) {
        if (count) std::memcpy(out, data + offset, count);
    }), borrowed_(data), nbytes_(size), parallel_reads_(true) {
    if (!data && size) throw std::invalid_argument("null NINT row storage");
    initialize();
}

NintRows::NintRows(std::size_t size, Read read, bool parallel_reads,ReadBatch batch)
    : read_(std::move(read)),read_batch_(std::move(batch)), nbytes_(size), parallel_reads_(parallel_reads) {
    if (!read_) throw std::invalid_argument("missing NINT row range reader");
    initialize();
}

const std::uint8_t* NintRows::selectors(
    std::size_t offset, std::size_t size, std::vector<std::uint8_t>& owned) {
    if (borrowed_) return borrowed_ + offset;
    owned.resize(size);
    read_(offset, owned.data(), size);
    return owned.data();
}

void NintRows::initialize() {
    Cursor cursor(nbytes_, read_);
    const auto raw_bits = cursor.scalar<std::uint8_t>();
    adaptive_ = mfq::nint_has_adaptive_storage(raw_bits);
    bits_ = mfq::nint_logical_bits(raw_bits);
    sub_bits_ = cursor.scalar<std::uint8_t>();
    group_size_ = cursor.scalar<std::int32_t>();
    const auto axis = cursor.scalar<std::int32_t>();
    width_ = cursor.scalar<std::int32_t>();
    const auto dimensions = cursor.scalar<std::uint32_t>();
    if (bits_ < 1 || bits_ > 8 || sub_bits_ < 1 || sub_bits_ > 8 ||
        group_size_ <= 0 || axis != 0 || width_ <= 0 || dimensions != 2) {
        throw std::runtime_error("invalid mapped NINT row geometry");
    }
    const auto shape_rows = cursor.scalar<std::int64_t>();
    const auto shape_width = cursor.scalar<std::int64_t>();
    const auto rows = cursor.scalar<std::uint32_t>();
    const auto groups = cursor.scalar<std::uint32_t>();
    if (rows == 0 || rows > std::numeric_limits<int>::max() ||
        shape_rows != rows || shape_width != width_ ||
        groups != (static_cast<std::uint64_t>(width_) + group_size_ - 1) / group_size_) {
        throw std::runtime_error("inconsistent mapped NINT row shape");
    }
    rows_ = static_cast<int>(rows);
    groups_ = static_cast<int>(groups);
    neuron_scale_offset_ = cursor.skip(static_cast<std::size_t>(rows) * 2);
    neuron_min_offset_ = cursor.skip(static_cast<std::size_t>(rows) * 2);
    const auto values_per_row = static_cast<std::uint64_t>(groups) * group_size_;
    if (adaptive_) {
        const auto size = packed_size(rows, 2);
        k_selectors_ = selectors(cursor.skip(size), size, owned_k_);
        const auto counts = build_ranks(k_selectors_, rows_, 2, k_ranks_);
        k_counts_=counts;
        for (int cohort = 0; cohort < 4; ++cohort) {
            const int width = sub_bits_ - 1 + cohort;
            if (counts[cohort] && (width < 1 || width > 8)) {
                throw std::runtime_error("invalid mapped NINT k selector");
            }
            const auto bytes = counts[cohort]
                ? packed_size(std::uint64_t(counts[cohort]) * groups, width) : 0;
            scale_offsets_[cohort] = cursor.skip(bytes);
            min_offsets_[cohort] = cursor.skip(bytes);
        }
        const auto qsize = packed_size(rows, 3);
        q_selectors_ = selectors(cursor.skip(qsize), qsize, owned_q_);
        const auto qcounts = build_ranks(q_selectors_, rows_, 3, q_ranks_);
        q_counts_=qcounts;
        for (int cohort = 0; cohort < 8; ++cohort) {
            q_offsets_[cohort] = cursor.skip(packed_size(
                std::uint64_t(qcounts[cohort]) * values_per_row, cohort + 1));
        }
    } else {
        const auto metadata_bytes = packed_size(std::uint64_t(rows) * groups, sub_bits_);
        scale_offsets_[0] = cursor.skip(metadata_bytes);
        min_offsets_[0] = cursor.skip(metadata_bytes);
        q_offsets_[0] = cursor.skip(packed_size(std::uint64_t(rows) * values_per_row, bits_));
    }
    if (!cursor.finished()) throw std::runtime_error("trailing mapped NINT row bytes");
}

std::size_t NintRows::index_nbytes() const noexcept {
    return k_ranks_.size() * sizeof(k_ranks_[0]) + q_ranks_.size() * sizeof(q_ranks_[0]) +
        owned_k_.size() + owned_q_.size();
}

std::vector<std::uint8_t> NintRows::slice_rows_blob(std::int64_t begin,std::int64_t end) const {
    if (begin<0 || end<=begin || end>rows_)
        throw std::out_of_range("NINT contiguous row range");
    const auto count=static_cast<std::uint32_t>(end-begin);
    std::vector<std::uint8_t> result;
    result.reserve(row_range_nbytes(begin,end));
    const auto scalar=[&](auto value) {
        const auto* bytes=reinterpret_cast<const std::uint8_t*>(&value);
        result.insert(result.end(),bytes,bytes+sizeof(value));
    };
    scalar(static_cast<std::uint8_t>(bits_|(adaptive_ ? kNintAdaptiveStorageFlag : 0)));
    scalar(static_cast<std::uint8_t>(sub_bits_));
    scalar(static_cast<std::int32_t>(group_size_)); scalar(std::int32_t(0));
    scalar(static_cast<std::int32_t>(width_)); scalar(std::uint32_t(2));
    scalar(static_cast<std::int64_t>(count)); scalar(static_cast<std::int64_t>(width_));
    scalar(count); scalar(static_cast<std::uint32_t>(groups_));
    for (auto base: {neuron_scale_offset_,neuron_min_offset_}) {
        const auto start=result.size(); result.resize(start+std::size_t(count)*2);
        read_(base+static_cast<std::size_t>(begin)*2,result.data()+start,std::size_t(count)*2);
    }
    const auto selected_bits=[&](const std::uint8_t* selectors,int bits) {
        detail::append_packed_range(result,[&](std::size_t offset,std::uint8_t* out,std::size_t size) {
            std::memcpy(out,selectors+offset,size);
        },0,static_cast<std::uint64_t>(begin)*bits,static_cast<std::uint64_t>(count)*bits);
    };
    if (adaptive_) selected_bits(k_selectors_,2);
    for (int cohort=0; cohort<(adaptive_ ? 4 : 1); ++cohort) {
        const auto rank=[&](std::int64_t row) -> std::uint32_t {
            if (!adaptive_) return static_cast<std::uint32_t>(row);
            return row==rows_ ? k_counts_[cohort] : cohort_rank(k_selectors_,
                static_cast<std::size_t>(row),2,cohort,k_ranks_);
        };
        const auto first=rank(begin), last=rank(end);
        if (first==last) continue;
        const int bits=adaptive_ ? sub_bits_-1+cohort : sub_bits_;
        const auto first_bit=std::uint64_t(first)*groups_*bits;
        const auto total_bits=std::uint64_t(last-first)*groups_*bits;
        detail::append_packed_range(result,read_,scale_offsets_[cohort],first_bit,total_bits);
        detail::append_packed_range(result,read_,min_offsets_[cohort],first_bit,total_bits);
    }
    if (adaptive_) selected_bits(q_selectors_,3);
    for (int cohort=0; cohort<(adaptive_ ? 8 : 1); ++cohort) {
        const auto rank=[&](std::int64_t row) -> std::uint32_t {
            if (!adaptive_) return static_cast<std::uint32_t>(row);
            return row==rows_ ? q_counts_[cohort] : cohort_rank(q_selectors_,
                static_cast<std::size_t>(row),3,cohort,q_ranks_);
        };
        const auto first=rank(begin), last=rank(end);
        const auto values=std::uint64_t(groups_)*group_size_;
        const int bits=adaptive_ ? cohort+1 : bits_;
        detail::append_packed_range(result,read_,q_offsets_[cohort],
            std::uint64_t(first)*values*bits,std::uint64_t(last-first)*values*bits);
    }
    return result;
}

std::size_t NintRows::row_range_nbytes(std::int64_t begin,std::int64_t end) const {
    if (begin<0 || end<=begin || end>rows_) throw std::out_of_range("NINT contiguous row range");
    const auto count=static_cast<std::uint64_t>(end-begin);
    std::size_t bytes=42+static_cast<std::size_t>(count)*4;
    if (adaptive_) bytes+=packed_size(count,2)+packed_size(count,3);
    for (int cohort=0; cohort<(adaptive_ ? 4 : 1); ++cohort) {
        const auto rank=[&](std::int64_t row) -> std::uint32_t {
            if (!adaptive_) return static_cast<std::uint32_t>(row);
            return row==rows_ ? k_counts_[cohort] : cohort_rank(k_selectors_,
                static_cast<std::size_t>(row),2,cohort,k_ranks_);
        };
        const auto rows=static_cast<std::uint64_t>(rank(end)-rank(begin));
        const int bits=adaptive_ ? sub_bits_-1+cohort : sub_bits_;
        if (rows) bytes+=2*packed_size(rows*groups_,bits);
    }
    for (int cohort=0; cohort<(adaptive_ ? 8 : 1); ++cohort) {
        const auto rank=[&](std::int64_t row) -> std::uint32_t {
            if (!adaptive_) return static_cast<std::uint32_t>(row);
            return row==rows_ ? q_counts_[cohort] : cohort_rank(q_selectors_,
                static_cast<std::size_t>(row),3,cohort,q_ranks_);
        };
        const auto rows=static_cast<std::uint64_t>(rank(end)-rank(begin));
        bytes+=packed_size(rows*groups_*group_size_,adaptive_ ? cohort+1 : bits_);
    }
    return bytes;
}

std::uint64_t NintRows::row_values_bits(std::int64_t begin,std::int64_t end) const {
    if (begin<0 || end<=begin || end>rows_) throw std::out_of_range("NINT contiguous row range");
    const auto prefix=[&](std::int64_t row) {
        if (!adaptive_) return std::uint64_t(row)*bits_;
        const auto& counts=row==rows_ ? q_counts_ : q_ranks_[static_cast<std::size_t>(row)/kRankStride];
        std::uint64_t sum=0;
        for (int cohort=0; cohort<8; ++cohort) sum+=std::uint64_t(counts[cohort])*(cohort+1);
        if (row!=rows_)
            for (auto index=static_cast<std::size_t>(row)-row%kRankStride; index<static_cast<std::size_t>(row); ++index)
                sum+=selector(q_selectors_,index,3)+1;
        return sum;
    };
    return (prefix(end)-prefix(begin))*groups_*group_size_;
}

void NintRows::append_row(std::int64_t row, NintRowBatch& batch) const {
    if (row < 0 || row >= rows_) throw std::out_of_range("mapped NINT row ID");
    if (batch.width_ && batch.width_ != width_) {
        throw std::runtime_error("NINT row batch widths disagree");
    }
    const auto local = static_cast<std::size_t>(row);
    const auto kc = adaptive_ ? selector(k_selectors_, local, 2) : 0;
    const auto qc = adaptive_ ? selector(q_selectors_, local, 3) : 0;
    const int k = adaptive_ ? sub_bits_ - 1 + static_cast<int>(kc) : sub_bits_;
    const int q = adaptive_ ? static_cast<int>(qc) + 1 : bits_;
    const auto kr = adaptive_ ? cohort_rank(k_selectors_, local, 2, kc, k_ranks_) : local;
    const auto qr = adaptive_ ? cohort_rank(q_selectors_, local, 3, qc, q_ranks_) : local;
    const auto kbit = std::uint64_t(kr) * groups_ * k;
    const auto qbit = std::uint64_t(qr) * groups_ * group_size_ * q;
    std::uint16_t scale, minimum;
    read_(neuron_scale_offset_ + local * 2, reinterpret_cast<std::uint8_t*>(&scale), 2);
    read_(neuron_min_offset_ + local * 2, reinterpret_cast<std::uint8_t*>(&minimum), 2);
    if ((scale & 0x7c00u) == 0x7c00u || (minimum & 0x7c00u) == 0x7c00u) {
        throw std::runtime_error("NINT row anchors must be finite");
    }
    const auto copy_stream = [&](std::size_t offset, std::uint64_t bit, std::uint64_t count, int bits) {
        const auto start = offset + static_cast<std::size_t>(bit / 8);
        const auto exact_bytes = static_cast<std::size_t>(((bit & 7) + count * bits + 7) / 8);
        if (start > nbytes_ || exact_bytes > nbytes_ - start ||
            exact_bytes >= std::numeric_limits<int>::max() ||
            batch.packed_.size() > std::numeric_limits<int>::max() - exact_bytes - 1) {
            throw std::runtime_error("NINT selected row stream exceeds bounds");
        }
        const auto destination = static_cast<std::uint32_t>(batch.packed_.size());
        batch.packed_.resize(batch.packed_.size() + exact_bytes);
        read_(start, batch.packed_.data() + destination, exact_bytes);
        batch.packed_.push_back(0); // The kernel may safely read a two-byte window.
        batch.source_bytes_read_ += exact_bytes;
        return destination;
    };
    const auto qoffset = copy_stream(q_offsets_[qc], qbit, std::uint64_t(groups_) * group_size_, q);
    const auto soffset = copy_stream(scale_offsets_[kc], kbit, groups_, k);
    const auto moffset = copy_stream(min_offsets_[kc], kbit, groups_, k);
    const std::uint32_t layout = q | (k << 4) | ((qbit & 7) << 8) |
        ((kbit & 7) << 12) | ((kbit & 7) << 16);
    batch.descriptors_.insert(batch.descriptors_.end(), {
        qoffset, soffset, moffset, layout, static_cast<std::uint32_t>(group_size_),
        std::uint32_t(scale) | (std::uint32_t(minimum) << 16)});
    batch.source_bytes_read_ += 4;
    batch.width_ = width_;
}

void NintRowBatch::append_batch(const NintRowBatch& other) {
    other.validate();
    if (this == &other || (width_ && width_ != other.width_)) {
        throw std::invalid_argument("NINT row batch merge disagrees");
    }
    if (other.packed_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        packed_.size() > std::numeric_limits<int>::max() - other.packed_.size() ||
        rows() + other.rows() > std::numeric_limits<int>::max() / 6 ||
        (rows() + other.rows()) * static_cast<std::uint64_t>(other.width_) >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("NINT merged row batch exceeds bounds");
    }
    const auto offset = static_cast<std::uint32_t>(packed_.size());
    // Preserve geometric vector growth when merging thousands of cached rows.
    packed_.insert(packed_.end(), other.packed_.begin(), other.packed_.end());
    for (std::size_t row = 0; row < other.rows(); ++row) {
        const auto* descriptor = other.descriptors_.data() + row * 6;
        descriptors_.insert(descriptors_.end(), {
            descriptor[0] + offset, descriptor[1] + offset, descriptor[2] + offset,
            descriptor[3], descriptor[4], descriptor[5]});
    }
    source_bytes_read_ += other.source_bytes_read_;
    width_ = other.width_;
}

void NintRows::append_rows(const std::int64_t* rows, std::size_t count,
        NintRowBatch& batch, int threads) const {
    if (!count) return;
    if (!rows) throw std::invalid_argument("null NINT row IDs");
    if ((batch.width_ && batch.width_ != width_) ||
        batch.rows() > std::numeric_limits<int>::max() / 6 ||
        count > std::numeric_limits<int>::max() / 6 - batch.rows() ||
        (batch.rows() + count) > std::numeric_limits<std::uint32_t>::max() / width_)
        throw std::runtime_error("NINT selected row batch exceeds bounds");
    for (std::size_t i = 0; i < count; ++i)
        if (rows[i] < 0 || rows[i] >= rows_) throw std::out_of_range("mapped NINT row ID");
    if(read_batch_) { append_rows_batched(rows,count,batch);return; }

    const auto workers = parallel_reads_ ? std::min<std::size_t>(count, std::max(threads, 1)) : 1;
    std::vector<NintRowBatch> pieces(workers);
    host_parallel_for(0, workers, 1, static_cast<int>(workers),
        [&](std::int64_t first, std::int64_t last) {
            for (auto part = first; part < last; ++part) {
                const auto begin = count * static_cast<std::size_t>(part) / workers;
                const auto end = count * static_cast<std::size_t>(part + 1) / workers;
                auto& piece = pieces[part];
                piece.descriptors_.reserve((end - begin) * 6);
                for (auto i = begin; i < end; ++i) append_row(rows[i], piece);
            }
        });
    // Publish only after all reads complete, including failure propagation.
    auto total = batch.packed_.size();
    if (total > std::numeric_limits<int>::max())
        throw std::runtime_error("NINT selected row stream exceeds bounds");
    for (const auto& piece : pieces) {
        if (piece.packed_.size() > std::numeric_limits<int>::max() - total)
            throw std::runtime_error("NINT selected row stream exceeds bounds");
        total += piece.packed_.size();
    }
    batch.packed_.reserve(total);
    batch.descriptors_.reserve((batch.rows() + count) * 6);
    for (const auto& piece : pieces) {
        const auto base = static_cast<std::uint32_t>(batch.packed_.size());
        batch.packed_.insert(batch.packed_.end(), piece.packed_.begin(), piece.packed_.end());
        for (std::size_t i = 0; i < piece.descriptors_.size(); i += 6)
            for (std::size_t field = 0; field < 6; ++field)
                batch.descriptors_.push_back(piece.descriptors_[i + field] + (field < 3 ? base : 0));
        batch.source_bytes_read_ += piece.source_bytes_read_;
    }
    batch.width_ = width_;
}

void NintRows::append_rows_batched(const int64_t* rows,std::size_t count,NintRowBatch& batch) const {
    NintRowBatch result;result.width_=width_;result.descriptors_.resize(count*6);
    struct Span {std::size_t source,size,destination;bool anchor;};
    std::vector<Span> plans;plans.reserve(count*5);
    for(std::size_t i=0;i<count;++i) {
        const auto row=static_cast<std::size_t>(rows[i]);
        const auto kc=adaptive_ ? selector(k_selectors_,row,2) : 0,qc=adaptive_ ? selector(q_selectors_,row,3) : 0;
        const int k=adaptive_ ? sub_bits_-1+int(kc) : sub_bits_,q=adaptive_ ? int(qc)+1 : bits_;
        const auto kr=adaptive_ ? cohort_rank(k_selectors_,row,2,kc,k_ranks_) : row;
        const auto qr=adaptive_ ? cohort_rank(q_selectors_,row,3,qc,q_ranks_) : row;
        const auto kbit=uint64_t(kr)*groups_*k,qbit=uint64_t(qr)*groups_*group_size_*q;
        const auto stream=[&](std::size_t start,uint64_t bit,uint64_t values,int bits) {
            start+=static_cast<std::size_t>(bit/8);
            const auto n=static_cast<std::size_t>(((bit&7)+values*bits+7)/8),offset=result.packed_.size();
            if(start>nbytes_ || n>nbytes_-start || n>=std::numeric_limits<int>::max() ||
                offset>std::numeric_limits<int>::max()-n-1)throw std::out_of_range("NINT batch stream exceeds bounds");
            plans.push_back({start,n,offset,false});result.packed_.resize(offset+n+1,0);result.source_bytes_read_+=n;
            return static_cast<uint32_t>(offset);
        };
        auto* words=result.descriptors_.data()+i*6;
        words[0]=stream(q_offsets_[qc],qbit,uint64_t(groups_)*group_size_,q);
        words[1]=stream(scale_offsets_[kc],kbit,groups_,k);words[2]=stream(min_offsets_[kc],kbit,groups_,k);
        words[3]=q | (k<<4) | ((qbit&7)<<8) | ((kbit&7)<<12) | ((kbit&7)<<16);
        words[4]=group_size_;
        plans.push_back({neuron_scale_offset_+row*2,2,(i*6+5)*sizeof(uint32_t),true});
        plans.push_back({neuron_min_offset_+row*2,2,(i*6+5)*sizeof(uint32_t)+2,true});result.source_bytes_read_+=4;
    }
    std::vector<ReadSpan> spans;spans.reserve(plans.size());
    for(const auto& p:plans)spans.push_back({p.source,reinterpret_cast<std::byte*>(p.anchor ?
        reinterpret_cast<uint8_t*>(result.descriptors_.data())+p.destination : result.packed_.data()+p.destination),p.size});
    read_batch_(spans);
    for(std::size_t i=0;i<count;++i) {
        const auto anchors=result.descriptors_[i*6+5];
        if((anchors&0x7c00u)==0x7c00u || ((anchors>>16)&0x7c00u)==0x7c00u)throw std::runtime_error("NINT row anchors must be finite");
    }
    const auto offset=batch.packed_.size();
    if(result.packed_.size()>std::numeric_limits<int>::max()-offset)throw std::overflow_error("NINT selected batch exceeds bounds");
    batch.packed_.reserve(offset+result.packed_.size());batch.descriptors_.reserve(batch.descriptors_.size()+count*6);
    batch.packed_.insert(batch.packed_.end(),result.packed_.begin(),result.packed_.end());
    for(std::size_t i=0;i<count*6;++i)batch.descriptors_.push_back(result.descriptors_[i]+(i%6<3 ? static_cast<uint32_t>(offset) : 0));
    batch.source_bytes_read_+=result.source_bytes_read_;batch.width_=width_;
}

void NintRowBatch::validate() const {
    if (rows() == 0 || rows() > std::numeric_limits<int>::max() / 6 ||
        rows() * static_cast<std::uint64_t>(width_) > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid NINT row decode batch size");
    }
}
void NintRowBatch::copy_row(std::size_t row,NintRowBatch& output) const {
    if(row>=rows())throw std::out_of_range("NINT row cache index");
    NintRowBatch result;result.width_=width_;
    const auto* d=descriptors_.data()+row*6;
    const int q=d[3]&15,k=(d[3]>>4)&15,group=d[4];const auto groups=(width_+group-1)/group;
    const std::size_t sizes[]={static_cast<std::size_t>((((d[3]>>8)&7)+uint64_t(groups)*group*q+7)/8),
        static_cast<std::size_t>((((d[3]>>12)&7)+uint64_t(groups)*k+7)/8),
        static_cast<std::size_t>((((d[3]>>16)&7)+uint64_t(groups)*k+7)/8)};
    for(int f=0;f<3;++f) {
        if(d[f]>packed_.size() || sizes[f]>packed_.size()-d[f])throw std::out_of_range("NINT cached row wire bounds");
        result.descriptors_.push_back(static_cast<uint32_t>(result.packed_.size()));
        result.packed_.insert(result.packed_.end(),packed_.begin()+d[f],packed_.begin()+d[f]+sizes[f]);result.packed_.push_back(0);
    }
    result.descriptors_.insert(result.descriptors_.end(),d+3,d+6);output.append_batch(result);
}
} // namespace mfq
