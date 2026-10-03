#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <functional>
#include <string_view>

namespace mfq::metal {

struct MlxPrefixCacheHooks {
    std::size_t block_size = 0;
    std::size_t input_tokens = 0;
    bool output_tokens = false;
    std::function<void(std::size_t)> capture;

    explicit operator bool() const noexcept { return block_size > 0 && bool(capture); }

    std::size_t prefill_end(std::size_t begin, std::size_t end) const noexcept {
        if (!*this) return end;
        end = std::min(end, (begin / block_size + 1) * block_size);
        if (begin < input_tokens) end = std::min(end, input_tokens);
        return end;
    }

    bool wants(std::size_t position, bool terminal = false) const noexcept {
        return *this && position > 0 && (output_tokens || position <= input_tokens) &&
            (terminal || position == input_tokens || position % block_size == 0);
    }

    void checkpoint(std::size_t position, bool terminal = false) const {
        if (wants(position, terminal)) capture(position);
    }

    int draft_limit(int position, int requested) const noexcept {
        if (!*this || !output_tokens) return requested;
        const auto remaining = block_size - static_cast<std::size_t>(position) % block_size;
        return std::min(requested, static_cast<int>(remaining - 1));
    }
};

inline std::size_t mlx_prefix_block_size(bool recurrent, std::size_t prefill_step,
    std::size_t physical_bytes, bool nax, bool wide_qsa) noexcept {
    if (!recurrent) return 256;
    const auto floor = physical_bytes >= 64ULL * 1024 * 1024 * 1024
        ? (nax ? (wide_qsa ? 8192 : 2048) : 4096) : 2048;
    return std::max<std::size_t>(floor, prefill_step);
}

inline bool mlx_prefix_nax_architecture(std::string_view architecture) noexcept {
    constexpr std::string_view prefix = "applegpu_g";
    if (!architecture.starts_with(prefix)) return false;
    architecture.remove_prefix(prefix.size());
    int generation = 0;
    const auto result = std::from_chars(architecture.data(), architecture.data() + architecture.size(), generation);
    return result.ec == std::errc{} && generation >= 17;
}

} // namespace mfq::metal
