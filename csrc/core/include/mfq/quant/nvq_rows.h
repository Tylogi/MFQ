#pragma once

#include <cstddef>
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>
#include "mfq/io/read_span.h"

namespace mfq {
// Index over a canonical NVQ/NPQ tensor. Only its compact header and
// codebook are retained; anchors and packed streams stay in the source.
class NvqRows {
public:
    using Read=std::function<void(std::size_t,std::uint8_t*,std::size_t)>;
    using ReadBatch=std::function<void(const std::vector<ReadSpan>&)>;
    NvqRows(std::size_t bytes,Read read,ReadBatch batch={});
    int rows() const noexcept { return rows_; }
    int width() const noexcept { return width_; }
    int format() const noexcept { return format_; }
    bool group64() const noexcept { return group64_; }
    bool byte_aligned_rows(std::int64_t rows) const noexcept;
    std::size_t index_nbytes() const noexcept { return prefix_.size(); }
    std::vector<std::uint8_t> slice_rows_blob(std::int64_t begin,std::int64_t end) const;
    struct Fields {
        std::vector<std::uint16_t> neuron_scale;
        std::vector<std::uint8_t> states,indices,aux;
        std::size_t source_nbytes=0;
        std::size_t nbytes() const noexcept {
            return neuron_scale.size()*2+states.size()+indices.size()+aux.size();
        }
    };
    // Reuse a validated pool's immutable codebook. Deinterleave group64 into
    // the same packed fields without constructing another complete NVQ blob.
    std::optional<Fields> slice_rows_fields(std::int64_t begin,std::int64_t end) const;
    struct FieldTarget {std::uint8_t* data=nullptr;std::size_t bytes=0;};
    // Anchor (FP16), states, indices and auxiliary bits. Unsupported physical
    // layouts return false before reading or touching any target.
    bool copy_rows_fields(std::int64_t begin,std::int64_t end,const std::array<FieldTarget,4>& targets,
                          std::size_t* source_bytes=nullptr) const;
    std::size_t row_range_nbytes(std::int64_t begin,std::int64_t end) const;
private:
    Read read_;
    ReadBatch read_batch_;
    std::vector<std::uint8_t> prefix_;
    std::size_t bytes_=0,anchor_=0,states_=0,indices_=0,aux_=0;
    int rows_=0,width_=0,groups_=0,vectors_=0,signs_=0;
    int format_=0,state_bits_=0,index_bits_=0,aux_bits_=0;
    bool group64_=false;
};
} // namespace mfq
