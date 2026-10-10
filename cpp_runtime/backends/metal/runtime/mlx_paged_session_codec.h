#pragma once

#include "mfq_paged_prefix_cache.h"
#include "mlx_minicpmo45.h"
#include "mlx_qwen35_causal_lm.h"
#include "mlx_qwen4_causal_lm.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace mfq::metal {

using MlxPagedPayload = mfq::cache::PagedPrefixPayload;

template <typename SessionState>
struct MlxPagedSessionCodec {
    static constexpr bool available = false;
};

template <>
struct MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState> {
    static constexpr bool available = true;
    static constexpr std::string_view name = "minicpmo45-qwen3-kv-v1";

    static std::vector<MlxPagedPayload> encode(
        const MlxMiniCPMO45TextSessionState& state,
        std::size_t block_size,
        std::size_t first_block = 0);
    static MlxPagedPayload encode_block(
        const MlxMiniCPMO45TextSessionState& state,
        std::size_t block_size,
        std::size_t block_index);
    static MlxMiniCPMO45TextSessionState decode(
        const std::vector<MlxPagedPayload>& payloads,
        const std::vector<std::int64_t>& tokens,
        std::size_t block_size);
};

template <>
struct MlxPagedSessionCodec<MlxQwen35TextSessionState> {
    static constexpr bool available = true;
    static constexpr bool refresh_final_block = true;
    static constexpr bool supports_tail_blocks = true;
    static constexpr std::string_view name = "qwen35-hybrid-kv-v3";

    static std::vector<MlxPagedPayload> encode(
        const MlxQwen35TextSessionState& state,
        std::size_t block_size,
        std::size_t first_block = 0);
    static MlxPagedPayload encode_block(
        const MlxQwen35TextSessionState& state,
        std::size_t block_size,
        std::size_t block_index);
    static MlxQwen35TextSessionState decode(
        const std::vector<MlxPagedPayload>& payloads,
        const std::vector<std::int64_t>& tokens,
        std::size_t block_size);
    static std::size_t decodable_blocks(
        const std::vector<MlxPagedPayload>& payloads);
    static bool has_exact_boundary(const MlxPagedPayload& payload);
    static bool has_mtp(const MlxPagedPayload& payload);
    static std::size_t token_count(const MlxPagedPayload& payload);
};

template <>
struct MlxPagedSessionCodec<MlxQwen4TextSessionState> {
    static constexpr bool available = true;
    static constexpr bool refresh_final_block = true;
    static constexpr bool supports_tail_blocks = true;
    static constexpr std::string_view name = "qwen4-hybrid-qsa-ple-v2";

    static MlxPagedPayload encode_block(const MlxQwen4TextSessionState& state,
        std::size_t block_size, std::size_t block_index);
    static MlxQwen4TextSessionState decode(const std::vector<MlxPagedPayload>& payloads,
        const std::vector<std::int64_t>& tokens, std::size_t block_size);
    static std::size_t decodable_blocks(const std::vector<MlxPagedPayload>& payloads);
    static bool has_exact_boundary(const MlxPagedPayload& payload);
    static bool has_mtp(const MlxPagedPayload& payload);
    static std::size_t token_count(const MlxPagedPayload& payload);
};

} // namespace mfq::metal
