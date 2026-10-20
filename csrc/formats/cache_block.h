#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#endif

namespace mfq::cache {
using BlockHash = std::array<std::uint8_t, 32>;
namespace detail {
constexpr std::string_view kHashDomain = "mfq-paged-prefix-v1";
constexpr std::array<std::uint32_t, 64> kShaConstants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

inline std::uint32_t rotate_right(std::uint32_t value, int count) {
    return (value >> count) | (value << (32 - count));
}

class Sha256State {
public:
    Sha256State()
        : state_{
              0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

    void update(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        total_bytes_ += size;
        while (size > 0) {
            const auto count = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, bytes, count);
            block_size_ += count;
            bytes += count;
            size -= count;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    BlockHash finish() {
        const std::uint64_t bit_count = total_bytes_ * 8;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56) {
            std::fill(block_.begin() + block_size_, block_.end(), 0);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + block_size_, block_.begin() + 56, 0);
        for (int index = 0; index < 8; ++index) {
            block_[63 - index] = static_cast<std::uint8_t>(
                bit_count >> (index * 8));
        }
        transform(block_.data());
        BlockHash digest{};
        for (std::size_t index = 0; index < state_.size(); ++index) {
            digest[index * 4] = static_cast<std::uint8_t>(state_[index] >> 24);
            digest[index * 4 + 1] =
                static_cast<std::uint8_t>(state_[index] >> 16);
            digest[index * 4 + 2] =
                static_cast<std::uint8_t>(state_[index] >> 8);
            digest[index * 4 + 3] =
                static_cast<std::uint8_t>(state_[index]);
        }
        return digest;
    }

private:
    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            words[index] =
                (static_cast<std::uint32_t>(block[index * 4]) << 24) |
                (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16) |
                (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8) |
                static_cast<std::uint32_t>(block[index * 4 + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const auto s0 = rotate_right(words[index - 15], 7) ^
                rotate_right(words[index - 15], 18) ^
                (words[index - 15] >> 3);
            const auto s1 = rotate_right(words[index - 2], 17) ^
                rotate_right(words[index - 2], 19) ^
                (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const auto sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                rotate_right(e, 25);
            const auto choose = (e & f) ^ (~e & g);
            const auto temp1 = h + sum1 + choose + kShaConstants[index] +
                words[index];
            const auto sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                rotate_right(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

template <typename T>
void hash_little_endian(Sha256State& state, T value) {
    std::array<std::uint8_t, sizeof(T)> bytes{};
    using Unsigned = std::make_unsigned_t<T>;
    const auto converted = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(converted >> (index * 8));
    }
    state.update(bytes.data(), bytes.size());
}

} // namespace detail

inline BlockHash sha256(const void* data, std::size_t size) {
#if defined(__APPLE__)
    CC_SHA256_CTX state;
    CC_SHA256_Init(&state);
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    while (size > 0) {
        const auto count = std::min<std::size_t>(
            size, std::numeric_limits<CC_LONG>::max());
        CC_SHA256_Update(
            &state, bytes, static_cast<CC_LONG>(count));
        bytes += count;
        size -= count;
    }
    BlockHash digest{};
    CC_SHA256_Final(digest.data(), &state);
    return digest;
#else
    detail::Sha256State state;
    state.update(data, size);
    return state.finish();
#endif
}

inline BlockHash sha256(std::string_view value) {
    return sha256(value.data(), value.size());
}

inline std::string block_hash_hex(const BlockHash& hash) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const auto byte : hash) {
        stream << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return stream.str();
}

inline BlockHash prefix_block_hash(
        const BlockHash& compatibility_hash,
        std::size_t block_size_tokens,
        const BlockHash& parent,
        const std::int64_t* token_ids,
        std::size_t token_count,
        std::string_view extra_key) {
        if (token_ids == nullptr || token_count == 0 ||
            token_count > block_size_tokens) {
            throw std::invalid_argument("invalid paged prefix token block");
        }
        detail::Sha256State state;
        state.update(detail::kHashDomain.data(), detail::kHashDomain.size());
        state.update(compatibility_hash.data(), compatibility_hash.size());
        state.update(parent.data(), parent.size());
        detail::hash_little_endian<std::uint64_t>(state, extra_key.size());
        state.update(extra_key.data(), extra_key.size());
        detail::hash_little_endian<std::uint64_t>(state, token_count);
        for (std::size_t index = 0; index < token_count; ++index) {
            detail::hash_little_endian<std::int64_t>(state, token_ids[index]);
        }
        return state.finish();
    }
} // namespace mfq::cache
