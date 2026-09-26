#include "token_constraint.h"

#include "tokenizer.h"

#include "chat.h"
#include "chat/common.h"
#include "mfq_grammar.h"
#include "mfq_text.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::engine {
namespace {

class GrammarConstraint {
public:
    GrammarConstraint(
            const MfqTokenizer& tokenizer,
            const common_chat_params& params)
        : vocab_(mfq_text_get_vocab(tokenizer.context())),
          vocab_size_(tokenizer.vocab_size()) {
        if (vocab_ == nullptr || params.grammar.empty()) {
            throw std::invalid_argument(
                "cannot create an empty chat-template grammar");
        }

        std::vector<std::string> trigger_patterns;
        std::vector<mfq_text_token> trigger_tokens;
        trigger_patterns.reserve(params.grammar_triggers.size());
        trigger_tokens.reserve(params.grammar_triggers.size());
        for (const auto& trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                    trigger_patterns.push_back(regex_escape(trigger.value));
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                    trigger_patterns.push_back(trigger.value);
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                    const auto& pattern = trigger.value;
                    trigger_patterns.push_back(
                        pattern.empty()
                            ? "^$"
                            : (pattern.front() == '^' ? "" : "^") +
                                pattern +
                                (pattern.back() == '$' ? "" : "$"));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                    trigger_tokens.push_back(trigger.token);
                    break;
                default:
                    throw std::runtime_error(
                        "unknown chat-template grammar trigger type");
            }
        }

        std::vector<const char*> trigger_pattern_ptrs;
        trigger_pattern_ptrs.reserve(trigger_patterns.size());
        for (const auto& pattern : trigger_patterns) {
            trigger_pattern_ptrs.push_back(pattern.c_str());
        }
        grammar_ = mfq_text_grammar_init_impl(
            vocab_, params.grammar.c_str(), "root", params.grammar_lazy,
            trigger_pattern_ptrs.data(), trigger_pattern_ptrs.size(),
            trigger_tokens.data(), trigger_tokens.size());
        if (grammar_ == nullptr) {
            throw std::runtime_error(
                "failed to initialize chat-template grammar");
        }
        if (!params.grammar_lazy && !params.generation_prompt.empty()) {
            for (const auto token : tokenizer.tokenize(
                     params.generation_prompt, true)) {
                mfq_text_grammar_accept_impl(
                    *grammar_, static_cast<mfq_text_token>(token));
            }
        }
    }

    ~GrammarConstraint() {
        if (grammar_ != nullptr) mfq_text_grammar_free_impl(grammar_);
    }

    GrammarConstraint(const GrammarConstraint&) = delete;
    GrammarConstraint& operator=(const GrammarConstraint&) = delete;

    std::shared_ptr<GrammarConstraint> clone() const {
        if (grammar_ == nullptr) {
            throw std::logic_error(
                "cannot clone an uninitialized chat-template grammar");
        }
        return std::shared_ptr<GrammarConstraint>(
            new GrammarConstraint(
                vocab_, vocab_size_,
                mfq_text_grammar_clone_impl(*grammar_)));
    }

    bool allows(std::int64_t token) {
        if (token < 0 || token >= vocab_size_) return false;
        mfq_text_token_data candidate{
            static_cast<mfq_text_token>(token), 0.0f, 0.0f};
        mfq_text_token_data_array candidates{&candidate, 1, -1, false};
        mfq_text_grammar_apply_impl(*grammar_, &candidates);
        return std::isfinite(candidate.logit);
    }

    void apply(float* logits, std::size_t count) {
        if (logits == nullptr ||
                count != static_cast<std::size_t>(vocab_size_)) {
            throw std::invalid_argument(
                "grammar logits do not match tokenizer vocabulary");
        }
        candidates_.resize(count);
        for (std::size_t index = 0; index < count; ++index) {
            candidates_[index] = {
                static_cast<mfq_text_token>(index), logits[index], 0.0f};
        }
        mfq_text_token_data_array candidates{
            candidates_.data(), candidates_.size(), -1, false};
        mfq_text_grammar_apply_impl(*grammar_, &candidates);
        bool has_candidate = false;
        for (std::size_t index = 0; index < count; ++index) {
            logits[index] = candidates_[index].logit;
            has_candidate = has_candidate || std::isfinite(logits[index]);
        }
        if (!has_candidate) {
            throw std::runtime_error(
                "chat-template grammar rejected every token");
        }
    }

    void accept(std::int64_t token) {
        if (token < 0 || token >= vocab_size_) {
            throw std::out_of_range(
                "grammar accepted token is out of range");
        }
        mfq_text_grammar_accept_impl(
            *grammar_, static_cast<mfq_text_token>(token));
    }

private:
    GrammarConstraint(
            const mfq_text_vocab* vocab,
            std::int32_t vocab_size,
            mfq_text_grammar* grammar)
        : vocab_(vocab), vocab_size_(vocab_size), grammar_(grammar) {
        if (vocab_ == nullptr || vocab_size_ <= 0 || grammar_ == nullptr) {
            if (grammar_ != nullptr) mfq_text_grammar_free_impl(grammar_);
            throw std::invalid_argument(
                "cannot clone an invalid chat-template grammar");
        }
    }

    const mfq_text_vocab* vocab_ = nullptr;
    std::int32_t vocab_size_ = 0;
    mfq_text_grammar* grammar_ = nullptr;
    std::vector<mfq_text_token_data> candidates_;
};

MfqTokenConstraintPtr wrap_constraint(
        std::shared_ptr<GrammarConstraint> implementation) {
    auto constraint = std::make_shared<MfqTokenConstraint>();
    constraint->allows = [implementation](std::int64_t token) {
        return implementation->allows(token);
    };
    constraint->apply = [implementation](float* logits, std::size_t count) {
        implementation->apply(logits, count);
    };
    constraint->accept = [implementation](std::int64_t token) {
        implementation->accept(token);
    };
    constraint->clone = [implementation] {
        return wrap_constraint(implementation->clone());
    };
    return constraint;
}

} // namespace

MfqTokenConstraintPtr make_chat_token_constraint(
        const MfqTokenizer& tokenizer,
        const common_chat_params& params) {
    if (params.grammar.empty()) return {};
    return wrap_constraint(
        std::make_shared<GrammarConstraint>(tokenizer, params));
}

} // namespace mfq::engine
