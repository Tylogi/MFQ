#pragma once

#include "tokenizer.h"
#include "mfq/tokenizer_identity.h"

namespace mfq::engine {

inline std::string tokenizer_identity_bytes(const MfqTokenizer& tokenizer) {
    const auto* vocabulary = mfq_text_get_vocab(tokenizer.context());
    return mfq::tokenizer_identity_bytes(tokenizer.vocab_size(), [&](std::int64_t id) {
        const auto* text = mfq_text_vocab_get_text(vocabulary, static_cast<mfq_text_token>(id));
        if (text == nullptr) throw std::runtime_error("tokenizer has no token text");
        return std::string(text);
    });
}

} // namespace mfq::engine
