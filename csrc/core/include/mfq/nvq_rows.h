#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace mfq {
// Index over a canonical NVQ/NPQ tensor. Only its compact header and
// codebook are retained; anchors and packed streams stay in the source.
class NvqRows {
public:
    using Read=std::function<void(std::size_t,std::uint8_t*,std::size_t)>;
    NvqRows(std::size_t bytes,Read read);
    int rows() const noexcept { return rows_; }
    int width() const noexcept { return width_; }
    int format() const noexcept { return format_; }
    bool group64() const noexcept { return group64_; }
    std::size_t index_nbytes() const noexcept { return prefix_.size(); }
    std::vector<std::uint8_t> slice_rows_blob(std::int64_t begin,std::int64_t end) const;
    std::size_t row_range_nbytes(std::int64_t begin,std::int64_t end) const;
private:
    Read read_;
    std::vector<std::uint8_t> prefix_;
    std::size_t bytes_=0,anchor_=0,states_=0,indices_=0,aux_=0;
    int rows_=0,width_=0,groups_=0,vectors_=0,signs_=0;
    int format_=0,state_bits_=0,index_bits_=0,aux_bits_=0;
    bool group64_=false;
};
} // namespace mfq
