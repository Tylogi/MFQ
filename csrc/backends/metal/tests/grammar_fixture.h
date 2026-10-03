#pragma once

#include "chat.h"
#include "gguf.h"
#include "token_constraint.h"
#include "tokenizer.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mfq::metal::test {

inline std::vector<uint8_t> tokenizer_bytes(int vocab) {
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_empty(), &gguf_free);
        std::vector<std::string> tokens;
        for (int token = 0; token < vocab - 1; ++token) {
            tokens.emplace_back(1, static_cast<char>('a' + token));
        }
        tokens.emplace_back("<eos>");
        std::vector<const char*> pointers;
        for (const auto& token : tokens) pointers.push_back(token.c_str());
        std::vector<int32_t> types(vocab, 1);
        types.back() = 3;
        gguf_set_val_str(metadata.get(), "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(metadata.get(), "tokenizer.ggml.pre", "gpt-2");
        gguf_set_arr_str(metadata.get(), "tokenizer.ggml.tokens", pointers.data(), pointers.size());
        gguf_set_arr_str(metadata.get(), "tokenizer.ggml.merges", nullptr, 0);
        gguf_set_arr_data(metadata.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32,
                          types.data(), types.size());
        gguf_set_val_u32(metadata.get(), "tokenizer.ggml.eos_token_id", vocab - 1);
        gguf_set_val_bool(metadata.get(), "tokenizer.ggml.add_bos_token", false);
        std::vector<uint8_t> bytes(gguf_get_meta_size(metadata.get()));
        gguf_get_meta_data(metadata.get(), bytes.data());
        return bytes;
}

// Exercise the real engine-owned grammar cursor, including independent clones.
inline MfqTokenConstraintPtr grammar_constraint(int vocab, const std::string& grammar) {
    static std::map<int, std::unique_ptr<mfq::engine::MfqTokenizer>> tokenizers;
    auto& tokenizer = tokenizers[vocab];
    if (!tokenizer) {
        tokenizer = std::make_unique<mfq::engine::MfqTokenizer>(tokenizer_bytes(vocab));
    }
    common_chat_params params;
    params.grammar = grammar;
    return mfq::engine::make_chat_token_constraint(*tokenizer, params);
}

} // namespace mfq::metal::test
