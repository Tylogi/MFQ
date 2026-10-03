#pragma once
#include "models/common/causal_forward.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace mfq::models::qwen4_exp {

struct NgramContext {
    int64_t batch = 0;
    std::vector<int64_t> tokens;
};
struct NgramHashes {
    std::vector<int64_t> ids;
    NgramContext next;
};

// Unsigned mixing wraps modulo 2^64, then vocabulary reduction uses the signed
// bit pattern. EOS starts a new segment, including across cached chunks.
inline NgramHashes ngram_hashes(const int64_t *source, int64_t b, int64_t t, int64_t ngram_,
                                int64_t heads_, int64_t eos_,
                                const std::vector<int64_t> &multipliers_,
                                const std::vector<int64_t> &offsets_,
                                const std::vector<int64_t> &vocab_, const NgramContext &context_,
                                bool cache) {
    require_model(
        b > 0 && t > 0 && ngram_ > 1 && heads_ > 0 && multipliers_.size() == size_t(ngram_) &&
            offsets_.size() == size_t((ngram_ - 1) * heads_) && vocab_.size() == offsets_.size(),
        "invalid ngram hash geometry");
    for (auto size : vocab_)
        require_model(size > 0, "invalid ngram hash vocabulary");
    const auto prefix = ngram_ - 1, length = prefix + t, nh = prefix * heads_;
    require_model(!cache || context_.batch != b || context_.tokens.size() == size_t(b * prefix),
                  "invalid ngram history");
    std::vector<int64_t> global(b * t * nh), history(b * length, eos_);
    for (int64_t bi = 0; bi < b; ++bi) {
        if (cache && context_.batch == b)
            std::copy_n(context_.tokens.data() + bi * prefix, prefix, history.data() + bi * length);
        std::copy_n(source + bi * t, t, history.data() + bi * length + prefix);
        int64_t segment = 0;
        for (int64_t token = 0; token < length; ++token) {
            uint64_t mixed = uint64_t(history[bi * length + token]) * uint64_t(multipliers_[0]);
            for (int64_t shift = 1; shift < ngram_; ++shift) {
                auto value = token - shift >= segment ? history[bi * length + token - shift] : eos_;
                mixed ^= uint64_t(value) * uint64_t(multipliers_[shift]);
                if (token >= prefix) {
                    int64_t signed_hash;
                    std::memcpy(&signed_hash, &mixed, sizeof(mixed));
                    for (int64_t head = 0; head < heads_; ++head) {
                        const auto h = (shift - 1) * heads_ + head;
                        auto remainder = signed_hash % vocab_[h];
                        if (remainder < 0)
                            remainder += vocab_[h];
                        global[(bi * t + token - prefix) * nh + h] = remainder + offsets_[h];
                    }
                }
            }
            if (history[bi * length + token] == eos_)
                segment = token + 1;
        }
    }
    if (cache) {
        NgramContext next;
        next.batch = b;
        next.tokens.resize(b * prefix);
        for (int64_t bi = 0; bi < b; ++bi)
            std::copy_n(history.data() + bi * length + t, prefix, next.tokens.data() + bi * prefix);
        return {std::move(global), std::move(next)};
    }
    return {std::move(global), context_};
}

} // namespace mfq::models::qwen4_exp
