#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mfq {

inline constexpr std::string_view tokenizer_identity_format =
    "mfq.tokenizer-vocabulary.v1";

template <typename TokenText>
std::string tokenizer_identity_bytes(std::int64_t vocabulary, TokenText token_text) {
    if (vocabulary < 1 || vocabulary > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid tokenizer identity vocabulary size");
    }
    std::string bytes(tokenizer_identity_format);
    bytes.push_back('\0');
    const auto append_u32 = [&](std::uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8) {
            bytes.push_back(static_cast<char>((value >> shift) & 0xff));
        }
    };
    append_u32(static_cast<std::uint32_t>(vocabulary));
    for (std::int64_t id = 0; id < vocabulary; ++id) {
        std::string token = token_text(id);
        if (token.empty()) token = "[EMPTY_" + std::to_string(id) + "]";
        if (token.find('\0') != std::string::npos ||
            token.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("invalid tokenizer identity token text");
        }
        append_u32(static_cast<std::uint32_t>(token.size()));
        bytes.append(token);
    }
    return bytes;
}

inline void validate_tokenizer_identity(
    std::string_view format, std::string_view reference, std::string_view candidate) {
    if (format.empty() || reference.empty()) {
        throw std::runtime_error(
            "logits manifest lacks tokenizer identity; regenerate the reference");
    }
    if (format != tokenizer_identity_format) {
        throw std::runtime_error("unsupported logits tokenizer identity format");
    }
    const auto is_digest = [](std::string_view value) {
        return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
    };
    if (!is_digest(reference) || !is_digest(candidate)) {
        throw std::runtime_error("invalid logits tokenizer identity SHA-256");
    }
    if (reference != candidate) {
        throw std::runtime_error("KLD reference/model tokenizer mapping mismatch");
    }
}

} // namespace mfq
