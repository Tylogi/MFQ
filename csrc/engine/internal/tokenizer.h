#pragma once

#include "mfq_text.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mfq::engine {

class MfqTokenizer {
public:
    explicit MfqTokenizer(const std::string& path);
    explicit MfqTokenizer(const std::vector<uint8_t>& gguf);
    ~MfqTokenizer();

    MfqTokenizer(const MfqTokenizer&) = delete;
    MfqTokenizer& operator=(const MfqTokenizer&) = delete;

    int32_t vocab_size() const;
    std::string chat_template() const;
    const mfq_text_context* context() const;
    int32_t bos_token() const;
    int32_t eos_token() const;
    int32_t eot_token() const;
    int32_t pad_token() const;
    bool add_bos() const;
    bool add_eos() const;
    std::vector<int64_t> tokenize(
        const std::string& text, bool parse_special, bool add_special = false) const;
    int64_t special_token_id(const std::string& text) const;
    bool is_eog(int64_t token) const;
    std::string piece(int64_t token, bool special = false) const;

private:
    void load_from_file(const std::string& path);
    void finish_init();

    mfq_text_context* context_ = nullptr;
    const mfq_text_vocab* vocab_ = nullptr;
};

} // namespace mfq::engine

struct MfqTokenizerProbe {
    int32_t vocab_size = 0;
    int32_t bos_token = -1;
    int32_t eos_token = -1;
    int32_t eot_token = -1;
    int32_t pad_token = -1;
    bool add_bos = false;
    bool add_eos = false;
    std::string chat_template;
    std::vector<int64_t> tokens;
};

MfqTokenizerProbe probe_mfq_tokenizer(
    const std::vector<uint8_t> & tokenizer_gguf,
    const std::string & text,
    bool add_special = false,
    bool parse_special = true);
MfqTokenizerProbe probe_mfq_tokenizer(
    const std::string & tokenizer_model,
    const std::string & text,
    bool add_special = false,
    bool parse_special = true);
