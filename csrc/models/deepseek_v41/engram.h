#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::models::deepseek_v41 {
inline constexpr int64_t kDeadToken = -1;
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
  public:
    EngramHashState(int vocabulary, int max_ngram_size, int heads, std::int64_t pad_id,
                    std::vector<std::int64_t> layer_ids, std::vector<std::int64_t> table_rows,
                    std::vector<std::int64_t> primes, std::vector<std::int64_t> offsets,
                    std::vector<std::int64_t> multipliers, std::vector<std::int32_t> token_map)
        : vocabulary_(vocabulary), max_ngram_size_(max_ngram_size), heads_(heads), pad_id_(pad_id),
          layer_ids_(std::move(layer_ids)), table_rows_(std::move(table_rows)),
          primes_(std::move(primes)), offsets_(std::move(offsets)),
          multipliers_(std::move(multipliers)), token_map_(std::move(token_map)) {}

    void reset(int batch) {
        if (batch <= 0) {
            throw std::invalid_argument("DeepSeek-V4.1 Engram batch must be positive");
        }
        batch_ = batch;
        position_ = 0;
        context_.assign(static_cast<std::size_t>(batch * (max_ngram_size_ - 1)), pad_id_);
        speculative_context_.reset();
        speculative_position_ = -1;
    }

    void begin_speculative() {
        if (batch_ <= 0 || position_ <= 0 || context_.empty() || speculative_context_.has_value()) {
            throw std::runtime_error("invalid DeepSeek-V4.1 Engram speculative checkpoint");
        }
        speculative_position_ = position_;
        speculative_context_ = context_;
    }

    void commit_speculative() noexcept {
        speculative_context_.reset();
        speculative_position_ = -1;
    }

    void rollback_speculative() {
        if (!speculative_context_.has_value() || speculative_position_ < 0) {
            throw std::runtime_error("DeepSeek-V4.1 Engram speculative checkpoint is unavailable");
        }
        context_ = std::move(*speculative_context_);
        position_ = speculative_position_;
        speculative_context_.reset();
        speculative_position_ = -1;
    }

    EngramHashBatch forward(const int64_t *ids, int batch, int tokens, int64_t pos0) {
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
            if (token < 0 || token >= static_cast<std::int64_t>(token_map_.size())) {
                throw std::out_of_range("DeepSeek-V4.1 Engram token ID is outside the vocabulary");
            }
            compressed[index] = token_map_[static_cast<std::size_t>(token)];
        }

        const int prefix = max_ngram_size_ - 1;
        const int combined_length = prefix + tokens;
        const int layers = static_cast<int>(layer_ids_.size());
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
                        static_cast<std::uint64_t>(multipliers_[multiplier_base]);
                    for (int shift = 1; shift < max_ngram_size_; ++shift) {
                        rolling ^=
                            static_cast<std::uint64_t>(ngram_tokens[shift]) *
                            static_cast<std::uint64_t>(multipliers_[multiplier_base + shift]);
                        std::int64_t signed_hash = 0;
                        std::memcpy(&signed_hash, &rolling, sizeof(signed_hash));
                        for (int head = 0; head < heads_; ++head) {
                            const int column = (shift - 1) * heads_ + head;
                            const auto bucket = static_cast<std::size_t>(layer * columns + column);
                            const auto prime = primes_[bucket];
                            auto remainder = signed_hash % prime;
                            if (remainder < 0)
                                remainder += prime;
                            const auto hash = remainder + offsets_[bucket];
                            if (hash < 0 || hash >= table_rows_[layer]) {
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
    std::vector<std::int64_t> layer_ids_;
    std::vector<std::int64_t> table_rows_;
    std::vector<std::int64_t> primes_;
    std::vector<std::int64_t> offsets_;
    std::vector<std::int64_t> multipliers_;
    std::vector<std::int32_t> token_map_;
    int batch_ = 0;
    std::int64_t position_ = 0;
    std::vector<std::int64_t> context_;
    std::optional<std::vector<std::int64_t>> speculative_context_;
    std::int64_t speculative_position_ = -1;
};

} // namespace mfq::models::deepseek_v41
