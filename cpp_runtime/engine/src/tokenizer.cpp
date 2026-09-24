#include "tokenizer.h"

#include "ggml.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace mfq::engine {
namespace {

void quiet_text_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::cerr << "mfq-text: " << text;
    }
}

} // namespace

MfqTokenizer::MfqTokenizer(const std::string& path) {
    load_from_file(path);
    finish_init();
}

MfqTokenizer::MfqTokenizer(const std::vector<uint8_t>& gguf) {
    if (gguf.empty()) {
        throw std::runtime_error("embedded tokenizer GGUF is empty");
    }
    ggml_log_set(quiet_text_log, nullptr);
    context_ = mfq_text_load_buffer(gguf.data(), gguf.size());
    if (context_ == nullptr) {
        throw std::runtime_error(
            "cannot initialize tokenizer from embedded GGUF metadata");
    }
    finish_init();
}

MfqTokenizer::~MfqTokenizer() {
    mfq_text_free(context_);
}

int32_t MfqTokenizer::vocab_size() const {
    return mfq_text_vocab_n_tokens(vocab_);
}

std::string MfqTokenizer::chat_template() const {
    const char* value = mfq_text_get_chat_template(context_, nullptr);
    return value == nullptr ? std::string() : std::string(value);
}

const mfq_text_context* MfqTokenizer::context() const {
    return context_;
}

int32_t MfqTokenizer::bos_token() const {
    return mfq_text_vocab_bos(vocab_);
}

int32_t MfqTokenizer::eos_token() const {
    return mfq_text_vocab_eos(vocab_);
}

int32_t MfqTokenizer::eot_token() const {
    return mfq_text_vocab_eot(vocab_);
}

int32_t MfqTokenizer::pad_token() const {
    return mfq_text_vocab_pad(vocab_);
}

bool MfqTokenizer::add_bos() const {
    return mfq_text_vocab_get_add_bos(vocab_);
}

bool MfqTokenizer::add_eos() const {
    return mfq_text_vocab_get_add_eos(vocab_);
}

std::vector<int64_t> MfqTokenizer::tokenize(
        const std::string& text, bool parse_special, bool add_special) const {
    int32_t n = mfq_text_tokenize(
        vocab_, text.data(), static_cast<int32_t>(text.size()),
        nullptr, 0, add_special, parse_special);
    if (n == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("tokenized prompt exceeds the tokenizer limit");
    }
    if (n == 0) return {};
    if (n > 0) {
        throw std::runtime_error("tokenizer returned an invalid sizing result");
    }
    std::vector<mfq_text_token> tokens(static_cast<size_t>(-n));
    n = mfq_text_tokenize(
        vocab_, text.data(), static_cast<int32_t>(text.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()),
        add_special, parse_special);
    if (n < 0) {
        throw std::runtime_error(
            "tokenizer buffer sizing changed unexpectedly");
    }
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        out.push_back(tokens[static_cast<size_t>(i)]);
    }
    return out;
}

int64_t MfqTokenizer::special_token_id(const std::string& text) const {
    const auto tokens = tokenize(text, true, false);
    if (tokens.size() != 1) {
        throw std::runtime_error(
            "tokenizer does not map the required special token to one ID: " + text);
    }
    return tokens.front();
}

bool MfqTokenizer::is_eog(int64_t token) const {
    return mfq_text_vocab_is_eog(vocab_, static_cast<mfq_text_token>(token));
}

std::string MfqTokenizer::piece(int64_t token, bool special) const {
    char local[128];
    int32_t n = mfq_text_token_to_piece(
        vocab_, static_cast<mfq_text_token>(token), local,
        static_cast<int32_t>(sizeof(local)), 0, special);
    if (n >= 0) return std::string(local, local + n);
    std::string out(static_cast<size_t>(-n), '\0');
    n = mfq_text_token_to_piece(
        vocab_, static_cast<mfq_text_token>(token), out.data(),
        static_cast<int32_t>(out.size()), 0, special);
    if (n < 0) {
        throw std::runtime_error("token piece buffer sizing changed unexpectedly");
    }
    out.resize(static_cast<size_t>(n));
    return out;
}

void MfqTokenizer::load_from_file(const std::string& path) {
    ggml_log_set(quiet_text_log, nullptr);
    context_ = mfq_text_load_file(path.c_str());
    if (context_ == nullptr) {
        throw std::runtime_error(
            "cannot load tokenizer metadata from GGUF: " + path);
    }
}

void MfqTokenizer::finish_init() {
    vocab_ = mfq_text_get_vocab(context_);
    if (vocab_ == nullptr) {
        mfq_text_free(context_);
        context_ = nullptr;
        throw std::runtime_error("GGUF does not contain a tokenizer vocabulary");
    }
}

} // namespace mfq::engine
