#pragma once
#include <cstddef>
#include <stdexcept>

namespace mfq {
inline size_t peer_first_parallel_launch_index(
        size_t launch_position,
        size_t rank_count,
        size_t primary_rank = 0) {
    if (rank_count == 0 || launch_position >= rank_count ||
            primary_rank >= rank_count) {
        throw std::runtime_error(
            "parallel launch position or primary rank is out of range");
    }
    if (rank_count == 1) {
        return 0;
    }
    if (launch_position == rank_count - 1) {
        return primary_rank;
    }
    return launch_position < primary_rank
        ? launch_position
        : launch_position + 1;
}

} // namespace mfq
