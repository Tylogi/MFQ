#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mfq::detail {
// Copy a contiguous bit interval without changing any encoded value. A
// non-byte-aligned interval is rebased to bit zero and its padding is zeroed.
// Reads are bounded by the exact source interval; no guard bytes are assumed.
template<class Read>
void append_packed_range(std::vector<std::uint8_t>& output,const Read& read,
        std::size_t stream,std::uint64_t bit,std::uint64_t count) {
    if (!count) return;
    const auto shift=static_cast<unsigned>(bit&7);
    if (count>std::numeric_limits<std::uint64_t>::max()-shift-7)
        throw std::overflow_error("packed row range is too large");
    const auto bytes64=(count+7)/8, source64=(count+shift+7)/8;
    if (source64>std::numeric_limits<std::size_t>::max() ||
        bit/8>std::numeric_limits<std::size_t>::max()-stream ||
        bytes64>output.max_size()-output.size())
        throw std::overflow_error("packed row range is too large");
    const auto offset=stream+static_cast<std::size_t>(bit/8);
    const auto bytes=static_cast<std::size_t>(bytes64), start=output.size();
    output.resize(start+bytes);
    if (!shift) read(offset,output.data()+start,bytes);
    else {
        std::vector<std::uint8_t> source(static_cast<std::size_t>(source64));
        read(offset,source.data(),source.size());
        for (std::size_t i=0; i<bytes; ++i) {
            std::uint16_t word=source[i];
            if (i+1<source.size()) word|=std::uint16_t(source[i+1])<<8;
            output[start+i]=static_cast<std::uint8_t>(word>>shift);
        }
    }
    if (count&7) output.back()&=static_cast<std::uint8_t>((1u<<(count&7))-1);
}
} // namespace mfq::detail
