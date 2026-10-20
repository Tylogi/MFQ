#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::models::deepseek_v41 {
inline constexpr int64_t kDeadToken = -1;
struct EngramHashSnapshot {
    int batch = 0;
    std::int64_t position = 0;
    std::vector<std::int64_t> context;
};
struct EngramHashBatch {
    int batch = 0;
    int tokens = 0;
    int layers = 0;
    int columns = 0;
    std::vector<std::int64_t> values;

    const std::int64_t *layer_data(int index) const {
        if (index < 0 || index >= layers || batch <= 0 || tokens <= 0 || columns <= 0) {
            throw std::out_of_range("invalid DeepSeek-V4.1 Engram hash layer");
        }
        const auto layer_size = static_cast<std::size_t>(batch) * static_cast<std::size_t>(tokens) *
                                static_cast<std::size_t>(columns);
        const auto begin = static_cast<std::size_t>(index) * layer_size;
        if (begin > values.size() || layer_size > values.size() - begin) {
            throw std::runtime_error("truncated DeepSeek-V4.1 Engram hash batch");
        }
        return values.data() + begin;
    }
};

class EngramHashState {
    struct Parameters {
        std::vector<std::int64_t> layer_ids,table_rows,primes,offsets,multipliers;
        std::vector<std::int32_t> token_map;
    };
  public:
    EngramHashState(int vocabulary, int max_ngram_size, int heads, std::int64_t pad_id,
                    std::vector<std::int64_t> layer_ids, std::vector<std::int64_t> table_rows,
                    std::vector<std::int64_t> primes, std::vector<std::int64_t> offsets,
                    std::vector<std::int64_t> multipliers, std::vector<std::int32_t> token_map)
        : vocabulary_(vocabulary), max_ngram_size_(max_ngram_size), heads_(heads), pad_id_(pad_id),
          parameters_(std::make_shared<const Parameters>(Parameters{
              std::move(layer_ids),std::move(table_rows),std::move(primes),std::move(offsets),
              std::move(multipliers),std::move(token_map)})) {}

    void reset(int batch) {
        if (batch <= 0) {
            throw std::invalid_argument("DeepSeek-V4.1 Engram batch must be positive");
        }
        batch_ = batch;
        position_ = 0;
        context_.assign(static_cast<std::size_t>(batch * (max_ngram_size_ - 1)), pad_id_);
        speculative_context_.reset();
        speculative_position_ = -1;
        speculative_tokens_.clear();
    }

    EngramHashSnapshot snapshot() const {
        if(batch_<=0 || position_<0 || speculative_context_)
            throw std::runtime_error("Engram session history is not committed");
        return {batch_,position_,context_};
    }

    void restore(const EngramHashSnapshot& state) {
        if(state.batch<=0 || state.position<0 ||
                state.context.size()!=std::size_t(state.batch)*std::size_t(max_ngram_size_-1) ||
                std::any_of(state.context.begin(),state.context.end(),[&](auto value) {
                    return value<kDeadToken || value>=vocabulary_;
                }))throw std::invalid_argument("Engram session history is incompatible");
        auto context=state.context;
        commit_speculative();
        batch_=state.batch;position_=state.position;context_.swap(context);
    }

    void begin_speculative() {
        if (batch_ <= 0 || position_ <= 0 || context_.empty() || speculative_context_.has_value()) {
            throw std::runtime_error("invalid DeepSeek-V4.1 Engram speculative checkpoint");
        }
        speculative_position_ = position_;
        speculative_context_ = context_;
        speculative_tokens_.clear();
    }

    void commit_speculative() noexcept {
        speculative_context_.reset();
        speculative_position_ = -1;
        speculative_tokens_.clear();
    }

    void rollback_speculative(std::int64_t position = -1) {
        if (!speculative_context_.has_value() || speculative_position_ < 0) {
            throw std::runtime_error("DeepSeek-V4.1 Engram speculative checkpoint is unavailable");
        }
        if(position<0)position=speculative_position_;
        if(position<speculative_position_ || position>position_)
            throw std::out_of_range("Engram retained prefix is outside its checkpoint");
        const auto kept=position-speculative_position_,drafted=position_-speculative_position_;
        if(kept && speculative_tokens_.size()!=std::size_t(batch_)*drafted)
            throw std::runtime_error("Engram speculative token history is incomplete");
        const int prefix=max_ngram_size_-1;
        for(int batch=0;batch<batch_;++batch)for(int column=0;column<prefix;++column) {
            const auto index=kept+column;
            context_[std::size_t(batch)*prefix+column]=index<prefix?
                (*speculative_context_)[std::size_t(batch)*prefix+index]:
                speculative_tokens_[std::size_t(batch)*drafted+index-prefix];
        }
        position_=position;
        commit_speculative();
    }

    // Only the small rolling history is copied. Immutable hash tables and the
    // vocabulary map are shared, so previews neither advance nor journal state.
    EngramHashBatch preview(const int64_t* ids,int batch,int tokens,int64_t pos0) const {
        auto next=*this;
        next.commit_speculative();
        return next.forward(ids,batch,tokens,pos0);
    }

    EngramHashBatch forward(const int64_t *ids, int batch, int tokens, int64_t pos0,
                           const std::uint8_t* image_mask = nullptr) {
        if (!ids || batch <= 0 || tokens <= 0 || pos0 < 0)
            throw std::invalid_argument("invalid Engram hash input");
        if (pos0 == 0)
            reset(batch);
        if (batch_ != batch || position_ != pos0 || context_.empty()) {
            throw std::runtime_error("DeepSeek-V4.1 Engram cache position disagrees");
        }
        const auto element_count = static_cast<std::size_t>(batch) * tokens;
        std::vector<std::int64_t> compressed(element_count);
        for (std::size_t index = 0; index < element_count; ++index) {
            const auto token = ids[index];
            if (token < 0 || token >= static_cast<std::int64_t>(parameters_->token_map.size())) {
                throw std::out_of_range("DeepSeek-V4.1 Engram token ID is outside the vocabulary");
            }
            compressed[index] = image_mask && image_mask[index] ? kDeadToken :
                parameters_->token_map[static_cast<std::size_t>(token)];
        }

        if(speculative_context_) {
            const auto previous=std::size_t(position_-speculative_position_);
            const auto count=previous+std::size_t(tokens);
            std::vector<std::int64_t> saved(std::size_t(batch)*count);
            for(int bi=0;bi<batch;++bi) {
                if(previous)std::copy_n(speculative_tokens_.data()+std::size_t(bi)*previous,
                    previous,saved.data()+std::size_t(bi)*count);
                std::copy_n(compressed.data()+std::size_t(bi)*tokens,tokens,
                    saved.data()+std::size_t(bi)*count+previous);
            }
            speculative_tokens_=std::move(saved);
        }

        const int prefix = max_ngram_size_ - 1;
        const int combined_length = prefix + tokens;
        const int layers = static_cast<int>(parameters_->layer_ids.size());
        const int columns = (max_ngram_size_ - 1) * heads_;
        EngramHashBatch result;
        result.batch = batch;
        result.tokens = tokens;
        result.layers = layers;
        result.columns = columns;
        result.values.resize(static_cast<std::size_t>(layers) * batch * tokens * columns);
        std::vector<std::int64_t> history(static_cast<std::size_t>(batch) * combined_length,
                                          pad_id_);
        for (int bi = 0; bi < batch; ++bi) {
            auto *row = history.data() + static_cast<std::size_t>(bi) * combined_length;
            std::copy_n(context_.data() + static_cast<std::size_t>(bi) * prefix, prefix, row);
            std::copy_n(compressed.data() + static_cast<std::size_t>(bi) * tokens, tokens,
                        row + prefix);
            for (int token = 0; token < tokens; ++token) {
                std::array<std::int64_t, 32> ngram_tokens{};
                if (max_ngram_size_ > static_cast<int>(ngram_tokens.size())) {
                    throw std::runtime_error(
                        "DeepSeek-V4.1 Engram ngram size exceeds native bound");
                }
                bool blocked = false;
                for (int shift = 0; shift < max_ngram_size_; ++shift) {
                    const auto source = row[prefix + token - shift];
                    blocked = blocked || source == kDeadToken;
                    ngram_tokens[static_cast<std::size_t>(shift)] = blocked ? pad_id_ : source;
                }
                for (int layer = 0; layer < layers; ++layer) {
                    const auto multiplier_base = static_cast<std::size_t>(layer * max_ngram_size_);
                    std::uint64_t rolling =
                        static_cast<std::uint64_t>(ngram_tokens[0]) *
                        static_cast<std::uint64_t>(parameters_->multipliers[multiplier_base]);
                    for (int shift = 1; shift < max_ngram_size_; ++shift) {
                        rolling ^=
                            static_cast<std::uint64_t>(ngram_tokens[shift]) *
                            static_cast<std::uint64_t>(parameters_->multipliers[multiplier_base + shift]);
                        std::int64_t signed_hash = 0;
                        std::memcpy(&signed_hash, &rolling, sizeof(signed_hash));
                        for (int head = 0; head < heads_; ++head) {
                            const int column = (shift - 1) * heads_ + head;
                            const auto bucket = static_cast<std::size_t>(layer * columns + column);
                            const auto prime = parameters_->primes[bucket];
                            auto remainder = signed_hash % prime;
                            if (remainder < 0)
                                remainder += prime;
                            const auto hash = remainder + parameters_->offsets[bucket];
                            if (hash < 0 || hash >= parameters_->table_rows[layer]) {
                                throw std::runtime_error(
                                    "DeepSeek-V4.1 Engram hash exceeds table rows");
                            }
                            const auto output =
                                ((static_cast<std::size_t>(layer) * batch + bi) * tokens + token) *
                                    columns +
                                column;
                            result.values[output] = hash;
                        }
                    }
                }
            }
        }
        for (int bi = 0; bi < batch; ++bi) {
            const auto *row = history.data() + static_cast<std::size_t>(bi) * combined_length;
            std::copy_n(row + tokens, prefix,
                        context_.data() + static_cast<std::size_t>(bi) * prefix);
        }
        position_ += tokens;
        return result;
    }

  private:
    int vocabulary_ = 0;
    int max_ngram_size_ = 0;
    int heads_ = 0;
    std::int64_t pad_id_ = 0;
    std::shared_ptr<const Parameters> parameters_;
    int batch_ = 0;
    std::int64_t position_ = 0;
    std::vector<std::int64_t> context_;
    std::optional<std::vector<std::int64_t>> speculative_context_;
    std::vector<std::int64_t> speculative_tokens_;
    std::int64_t speculative_position_ = -1;
};

} // namespace mfq::models::deepseek_v41
