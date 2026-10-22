#include "tokenizer_identity.h"
#include "gguf.h"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Action>
void rejects(Action action, const char* message) {
    try { action(); }
    catch (const std::runtime_error& error) {
        require(std::string(error.what()).find(message) != std::string::npos,
                "wrong tokenizer identity failure");
        return;
    }
    throw std::runtime_error("invalid tokenizer identity was accepted");
}

std::vector<std::uint8_t> fixture(const std::vector<std::string>& names) {
    std::vector<const char*> tokens;
    for (const auto& name : names) tokens.push_back(name.c_str());
    std::vector<std::int32_t> types(names.size(), MFQ_TEXT_TOKEN_TYPE_BYTE);
    types[0] = MFQ_TEXT_TOKEN_TYPE_UNKNOWN;
    types[1] = types[2] = MFQ_TEXT_TOKEN_TYPE_CONTROL;
    types.back() = MFQ_TEXT_TOKEN_TYPE_NORMAL;
    std::vector<float> scores(names.size());
    auto* metadata = gguf_init_empty();
    gguf_set_val_str(metadata, "general.architecture", "mfq-test");
    gguf_set_val_str(metadata, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(metadata, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_data(metadata, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(metadata, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_u32(metadata, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(metadata, "tokenizer.ggml.eos_token_id", 2);
    std::vector<std::uint8_t> data(gguf_get_meta_size(metadata));
    gguf_get_meta_data(metadata, data.data());
    gguf_free(metadata);
    return data;
}

} // namespace

int main() try {
    std::vector<std::string> golden_tokens{"<unk>", "\xc4\xa0hello", "\xe4\xbd\xa0\xe5\xa5\xbd", "[PAD3]"};
    const auto golden = mfq::tokenizer_identity_bytes(4, [&](auto id) { return golden_tokens[id]; });
    const char expected[] = "mfq.tokenizer-vocabulary.v1\0\x04\0\0\0"
        "\x05\0\0\0<unk>\x07\0\0\0\xc4\xa0hello"
        "\x06\0\0\0\xe4\xbd\xa0\xe5\xa5\xbd\x06\0\0\0[PAD3]";
    require(golden == std::string(expected, sizeof(expected) - 1), "canonical mapping encoding differs");

    std::vector<std::string> names{"<unk>", "<s>", "</s>"};
    for (int value = 0; value < 256; ++value) {
        char name[7];
        std::snprintf(name, sizeof(name), "<0x%02X>", value);
        names.emplace_back(name);
    }
    names.emplace_back("\xe4\xbd\xa0\xe5\xa5\xbd");
    mfq::engine::MfqTokenizer reference(fixture(names));
    const auto mapping = mfq::engine::tokenizer_identity_bytes(reference);
    require(mapping == mfq::tokenizer_identity_bytes(names.size(), [&](auto id) { return names[id]; }),
            "embedded tokenizer changed the token-ID mapping");
    std::swap(names[100], names[101]);
    mfq::engine::MfqTokenizer candidate(fixture(names));
    require(reference.vocab_size() == candidate.vocab_size(), "fixture vocabulary size changed");
    require(mapping != mfq::engine::tokenizer_identity_bytes(candidate), "reordered mapping was accepted");

    const std::string digest(64, 'a');
    mfq::validate_tokenizer_identity(mfq::tokenizer_identity_format, digest, digest);
    rejects([&] { mfq::validate_tokenizer_identity("", "", digest); }, "regenerate the reference");
    rejects([&] { mfq::validate_tokenizer_identity("unknown", digest, digest); }, "unsupported");
    rejects([&] { mfq::validate_tokenizer_identity(mfq::tokenizer_identity_format, "bad", digest); }, "SHA-256");
    rejects([&] { mfq::validate_tokenizer_identity(mfq::tokenizer_identity_format, digest, std::string(64, 'b')); }, "mapping mismatch");
    std::cout << "tokenizer identity tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
