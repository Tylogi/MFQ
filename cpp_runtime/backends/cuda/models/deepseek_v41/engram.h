#pragma once

#include "storage/weight_loader.h"

#include "mfe_expert_store.h"
#include "models/deepseek_v41/config.h"
#include "cpp_runtime/models/deepseek_v41/engram.h"
#include "models/deepseek_v41/causal_lm.h"
#include "quant_linear.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::cuda::deepseek_v41_runtime {

using EngramTensor = mfq_tensor_backend::Tensor;
using EngramConfig = mfq::models::deepseek_v41::Config;

namespace engram_detail {

constexpr std::array<std::uint8_t, 8> kMagic{'M', 'F', 'Q', 'E', 'N', 'G', 'R', '1'};
constexpr std::size_t kMxHeaderBytes = 56;
constexpr std::int64_t kDeadToken = -1;

class ByteCursor {
  public:
    explicit ByteCursor(const std::vector<std::uint8_t> &bytes) : bytes_(bytes) {}

    const std::uint8_t *take(std::size_t count, std::string_view description) {
        if (offset_ > bytes_.size() || count > bytes_.size() - offset_) {
            throw std::runtime_error("truncated DeepSeek-V4.1 Engram " + std::string(description));
        }
        const auto *result = bytes_.data() + offset_;
        offset_ += count;
        return result;
    }

    std::uint32_t u32(std::string_view description) {
        const auto *bytes = take(4, description);
        std::uint32_t result = 0;
        for (int index = 0; index < 4; ++index) {
            result |= static_cast<std::uint32_t>(bytes[index]) << (8 * index);
        }
        return result;
    }

    std::uint64_t u64(std::string_view description) {
        const auto *bytes = take(8, description);
        std::uint64_t result = 0;
        for (int index = 0; index < 8; ++index) {
            result |= static_cast<std::uint64_t>(bytes[index]) << (8 * index);
        }
        return result;
    }

    std::int32_t i32(std::string_view description) {
        const auto raw = u32(description);
        std::int32_t result = 0;
        std::memcpy(&result, &raw, sizeof(result));
        return result;
    }

    std::int64_t i64(std::string_view description) {
        const auto raw = u64(description);
        std::int64_t result = 0;
        std::memcpy(&result, &raw, sizeof(result));
        return result;
    }

    void require_end() const {
        if (offset_ != bytes_.size()) {
            throw std::runtime_error("DeepSeek-V4.1 Engram asset has trailing bytes");
        }
    }

  private:
    const std::vector<std::uint8_t> &bytes_;
    std::size_t offset_ = 0;
};

template <typename T, typename Reader>
std::vector<T> read_vector(ByteCursor &cursor, std::size_t count, Reader &&reader,
                           std::string_view description) {
    std::vector<T> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back(reader(cursor, description));
    }
    return result;
}

inline std::uint64_t checked_product(std::uint64_t left, std::uint64_t right,
                                     std::string_view description) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::runtime_error("DeepSeek-V4.1 Engram " + std::string(description) +
                                 " byte size overflows");
    }
    return left * right;
}

inline std::uint64_t read_u64_at(const std::vector<std::uint8_t> &bytes, std::size_t offset) {
    if (offset > bytes.size() || 8 > bytes.size() - offset) {
        throw std::runtime_error("truncated DeepSeek-V4.1 Engram MX header");
    }
    std::uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(bytes[offset + index]) << (8 * index);
    }
    return value;
}

inline float decode_e4m3(std::uint8_t raw) {
    if ((raw & 0x7fu) == 0x7fu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float sign = (raw & 0x80u) ? -1.0f : 1.0f;
    const int exponent = (raw >> 3u) & 0x0fu;
    const int mantissa = raw & 0x07u;
    return sign * (exponent == 0
                       ? std::ldexp(static_cast<float>(mantissa), -9)
                       : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7));
}

} // namespace engram_detail

using EngramHashBatch = mfq::models::deepseek_v41::EngramHashBatch;
class EngramHashState : public mfq::models::deepseek_v41::EngramHashState {
    using HashState = mfq::models::deepseek_v41::EngramHashState;

  public:
    using HashState::HashState;
    static std::unique_ptr<EngramHashState> load(const mfq::ModelSource &model,
                                                 const EngramConfig &config) {
        constexpr const char *asset_name = "__mfq_asset__/deepseek-v41-engram-v1.bin";
        if (!model.has_asset(asset_name)) {
            throw std::runtime_error("DeepSeek-V4.1 model source has no Engram hash asset");
        }
        const auto asset = read_asset(model, asset_name);
        engram_detail::ByteCursor cursor(asset);
        const auto *magic = cursor.take(engram_detail::kMagic.size(), "asset magic");
        if (!std::equal(engram_detail::kMagic.begin(), engram_detail::kMagic.end(), magic)) {
            throw std::runtime_error("invalid DeepSeek-V4.1 Engram asset magic");
        }
        const auto vocabulary = cursor.u32("vocabulary");
        const auto compressed = cursor.u32("compressed vocabulary");
        const auto layer_count = cursor.u32("layer count");
        const auto max_ngram = cursor.u32("maximum ngram size");
        const auto heads = cursor.u32("head count");
        const auto pad_id = cursor.i32("compressed pad token");
        if (vocabulary > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            compressed > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            layer_count == 0 ||
            layer_count > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            max_ngram < 2 ||
            max_ngram > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) || heads == 0 ||
            heads > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("invalid DeepSeek-V4.1 Engram asset dimensions");
        }
        const auto layers = static_cast<std::size_t>(layer_count);
        const auto orders = static_cast<std::size_t>(max_ngram - 1);
        const auto head_count = static_cast<std::size_t>(heads);
        const auto bucket_count = engram_detail::checked_product(
            engram_detail::checked_product(layers, orders, "bucket"), head_count, "bucket");
        const auto multiplier_count = engram_detail::checked_product(
            layers, static_cast<std::size_t>(max_ngram), "multiplier");
        auto layer_ids = engram_detail::read_vector<std::int64_t>(
            cursor, layers,
            [](auto &value, auto what) { return static_cast<std::int64_t>(value.i32(what)); },
            "layer ids");
        auto table_rows = engram_detail::read_vector<std::int64_t>(
            cursor, layers, [](auto &value, auto what) { return value.i64(what); }, "table rows");
        auto primes = engram_detail::read_vector<std::int64_t>(
            cursor, static_cast<std::size_t>(bucket_count),
            [](auto &value, auto what) { return value.i64(what); }, "bucket primes");
        auto offsets = engram_detail::read_vector<std::int64_t>(
            cursor, static_cast<std::size_t>(bucket_count),
            [](auto &value, auto what) { return value.i64(what); }, "bucket offsets");
        auto multipliers = engram_detail::read_vector<std::int64_t>(
            cursor, static_cast<std::size_t>(multiplier_count),
            [](auto &value, auto what) { return value.i64(what); }, "hash multipliers");
        auto token_map = engram_detail::read_vector<std::int32_t>(
            cursor, vocabulary, [](auto &value, auto what) { return value.i32(what); },
            "compressed token map");
        cursor.require_end();

        if (vocabulary != static_cast<std::uint32_t>(config.vocab) ||
            compressed != static_cast<std::uint32_t>(config.engram_compressed_vocab_size) ||
            max_ngram != static_cast<std::uint32_t>(config.engram_max_ngram_size) ||
            heads != static_cast<std::uint32_t>(config.engram_n_heads) ||
            layer_ids != config.engram_layer_ids || table_rows != config.engram_num_embeddings ||
            pad_id < 0 || pad_id >= static_cast<std::int64_t>(compressed)) {
            throw std::runtime_error("DeepSeek-V4.1 Engram asset disagrees with model config");
        }
        for (std::size_t layer = 0; layer < layers; ++layer) {
            const auto last = layer * orders * head_count + orders * head_count - 1;
            if (primes[last] <= 0 || offsets[last] < 0 ||
                primes[last] + offsets[last] != table_rows[layer]) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 Engram bucket layout disagrees with table rows");
            }
        }
        if (std::any_of(token_map.begin(), token_map.end(),
                        [compressed](std::int32_t value) {
                            return value < 0 || value >= static_cast<std::int32_t>(compressed);
                        }) ||
            std::any_of(multipliers.begin(), multipliers.end(),
                        [](std::int64_t value) { return value <= 0 || !(value & 1); })) {
            throw std::runtime_error("DeepSeek-V4.1 Engram hash metadata is invalid");
        }
        return std::unique_ptr<EngramHashState>(new EngramHashState(
            static_cast<int>(vocabulary), static_cast<int>(max_ngram), static_cast<int>(heads),
            pad_id, std::move(layer_ids), std::move(table_rows), std::move(primes),
            std::move(offsets), std::move(multipliers), std::move(token_map)));
    }

    EngramHashBatch forward(const EngramTensor &token_ids, std::int64_t pos0) {
        if (token_ids.dim() != 2 || token_ids.size(0) <= 0 || token_ids.size(1) <= 0 || pos0 < 0 ||
            token_ids.size(0) > std::numeric_limits<int>::max() ||
            token_ids.size(1) > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 Engram token IDs must have nonempty [B,T] shape");
        }
        const int batch = static_cast<int>(token_ids.size(0));
        const int tokens = static_cast<int>(token_ids.size(1));
        auto host_ids =
            token_ids.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        return HashState::forward(host_ids.data_ptr<int64_t>(), batch, tokens, pos0);
    }

    static void self_check() {
        const std::vector<std::int64_t> expected{
            3, 8, 15, 26, 3, 8, 15, 26, 3, 8, 16, 27, 3, 8, 14, 23,
        };
        const auto make_state = [] {
            return EngramHashState(8, 3, 2, 0, {1}, {36}, {5, 7, 11, 13}, {0, 5, 12, 23}, {3, 5, 7},
                                   {0, 1, 2, 3, 4, 5, 6, 7});
        };
        const auto options = mfq_tensor_backend::TensorOptions()
                                 .device(mfq_tensor_backend::kCPU)
                                 .dtype(mfq_tensor_backend::kInt64);
        auto all_ids = mfq_tensor_backend::tensor(std::vector<std::int64_t>{1, 2, 3, 4}, options)
                           .reshape({1, 4});
        auto full_state = make_state();
        const auto full = full_state.forward(all_ids, 0);
        MFQ_RUNTIME_CHECK(full.values == expected,
                          "DeepSeek-V4.1 Engram full hash contract failed");

        auto chunked_state = make_state();
        const auto first = chunked_state.forward(all_ids.narrow(1, 0, 2), 0);
        const auto second = chunked_state.forward(all_ids.narrow(1, 2, 2), 2);
        std::vector<std::int64_t> chunked = first.values;
        chunked.insert(chunked.end(), second.values.begin(), second.values.end());
        MFQ_RUNTIME_CHECK(chunked == expected, "DeepSeek-V4.1 Engram chunked hash contract failed");

        auto speculative_state = make_state();
        (void)speculative_state.forward(all_ids.narrow(1, 0, 2), 0);
        speculative_state.begin_speculative();
        const auto speculative = speculative_state.forward(all_ids.narrow(1, 2, 2), 2);
        speculative_state.rollback_speculative();
        const auto replayed = speculative_state.forward(all_ids.narrow(1, 2, 2), 2);
        MFQ_RUNTIME_CHECK(speculative.values == replayed.values,
                          "DeepSeek-V4.1 Engram speculative rollback failed");
        MFQ_RUNTIME_CHECK(engram_detail::decode_e4m3(0x00) == 0.0f &&
                              engram_detail::decode_e4m3(0x38) == 1.0f &&
                              engram_detail::decode_e4m3(0x40) == 2.0f &&
                              engram_detail::decode_e4m3(0xb8) == -1.0f &&
                              engram_detail::decode_e4m3(0x01) == std::ldexp(1.0f, -9),
                          "DeepSeek-V4.1 Engram E4M3 decode contract failed");
    }
};

class EngramTable {
  public:
    static std::shared_ptr<EngramTable> load(const mfq::ModelSource &model, const std::string &name,
                                             std::int64_t expected_rows, int expected_width,
                                             std::size_t cache_capacity_rows) {
        const auto record = require_tensor(model, name);
        if (record.dtype != "MXFP8") {
            throw std::runtime_error("DeepSeek-V4.1 Engram table must use native MXFP8: " + name);
        }
        auto header =
            read_record_range(model, name, record.nbytes, 0, engram_detail::kMxHeaderBytes);
        if (header.size() < engram_detail::kMxHeaderBytes ||
            std::memcmp(header.data(), "MXT1", 4) != 0 || header[4] != 1 || header[5] != 8 ||
            header[6] != 0 || header[7] != 0) {
            throw std::runtime_error("invalid DeepSeek-V4.1 Engram MXFP8 header: " + name);
        }
        const auto rows = engram_detail::read_u64_at(header, 8);
        const auto columns = engram_detail::read_u64_at(header, 16);
        const auto storage_rows = engram_detail::read_u64_at(header, 24);
        const auto storage_columns = engram_detail::read_u64_at(header, 32);
        const auto scale_rows = engram_detail::read_u64_at(header, 40);
        const auto scale_columns = engram_detail::read_u64_at(header, 48);
        if (rows != static_cast<std::uint64_t>(expected_rows) ||
            columns != static_cast<std::uint64_t>(expected_width) || storage_rows != rows ||
            storage_columns != columns || scale_rows != rows || scale_columns != columns / 32 ||
            columns == 0 || columns % 32) {
            throw std::runtime_error("DeepSeek-V4.1 Engram table is not row-wise MXFP8: " + name);
        }
        const auto value_bytes = engram_detail::checked_product(rows, columns, "value");
        const auto scale_bytes = engram_detail::checked_product(scale_rows, scale_columns, "scale");
        if (value_bytes >
                std::numeric_limits<std::uint64_t>::max() - engram_detail::kMxHeaderBytes ||
            scale_bytes > std::numeric_limits<std::uint64_t>::max() -
                              engram_detail::kMxHeaderBytes - value_bytes ||
            engram_detail::kMxHeaderBytes + value_bytes + scale_bytes != record.nbytes) {
            throw std::runtime_error("DeepSeek-V4.1 Engram MXFP8 payload size disagrees: " + name);
        }
        return std::shared_ptr<EngramTable>(
            new EngramTable(model, name, record.nbytes, static_cast<std::int64_t>(rows),
                            static_cast<int>(columns), value_bytes, cache_capacity_rows));
    }

    EngramTensor gather(const std::int64_t *rows, int batch, int tokens, int hash_columns,
                        const mfq_tensor_backend::Device &device) const {
        const auto expected = static_cast<std::size_t>(batch) * tokens * hash_columns;
        if (batch <= 0 || tokens <= 0 || hash_columns <= 0 || rows == nullptr) {
            throw std::invalid_argument("DeepSeek-V4.1 Engram lookup geometry disagrees");
        }
        std::vector<std::pair<std::int64_t, std::size_t>> requests;
        requests.reserve(expected);
        for (std::size_t index = 0; index < expected; ++index) {
            if (rows[index] < 0 || rows[index] >= rows_) {
                throw std::out_of_range("DeepSeek-V4.1 Engram lookup row is outside the table");
            }
            requests.emplace_back(rows[index], index);
        }
        std::sort(requests.begin(), requests.end(), [](const auto &left, const auto &right) {
            return left.first < right.first ||
                   (left.first == right.first && left.second < right.second);
        });
        std::vector<float> output(expected * static_cast<std::size_t>(width_));
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t begin = 0;
        while (begin < requests.size()) {
            std::size_t end = begin + 1;
            while (end < requests.size() && requests[end].first == requests[begin].first) {
                ++end;
            }
            const auto &decoded = row_locked(requests[begin].first);
            for (std::size_t request = begin; request < end; ++request) {
                std::copy_n(decoded.data(), width_,
                            output.data() +
                                requests[request].second * static_cast<std::size_t>(width_));
            }
            begin = end;
        }
        return mfq_tensor_backend::from_blob(
                   output.data(), {batch, tokens, static_cast<std::int64_t>(hash_columns) * width_},
                   mfq_tensor_backend::TensorOptions()
                       .device(mfq_tensor_backend::kCPU)
                       .dtype(mfq_tensor_backend::kFloat32))
            .clone()
            .to(device, mfq_tensor_backend::kFloat16)
            .contiguous();
    }

  private:
    struct CacheEntry {
        std::vector<float> values;
        std::list<std::int64_t>::iterator recency;
    };

    static std::vector<std::uint8_t>
    read_record_range(const mfq::ModelSource &source, const std::string &name, std::uint64_t nbytes,
                      std::uint64_t relative_offset, std::uint64_t count) {
        if (relative_offset > nbytes || count > nbytes - relative_offset ||
            count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::out_of_range("model tensor byte range is out of bounds: " + name);
        }
        std::vector<std::uint8_t> result(static_cast<std::size_t>(count));
        source.read_range_into(name, relative_offset, reinterpret_cast<std::byte *>(result.data()),
                               result.size());
        return result;
    }

    EngramTable(const mfq::ModelSource &source, std::string name, std::uint64_t nbytes,
                std::int64_t rows, int width, std::uint64_t value_bytes, std::size_t capacity)
        : source_(&source), name_(std::move(name)), nbytes_(nbytes), rows_(rows), width_(width),
          value_bytes_(value_bytes), capacity_(capacity) {
        for (std::size_t raw = 0; raw < e4m3_.size(); ++raw) {
            e4m3_[raw] = engram_detail::decode_e4m3(static_cast<std::uint8_t>(raw));
        }
        for (std::size_t raw = 0; raw < e8m0_.size(); ++raw) {
            e8m0_[raw] = raw == 255 ? std::numeric_limits<float>::quiet_NaN()
                                    : std::ldexp(1.0f, static_cast<int>(raw) - 127);
        }
        scratch_.resize(static_cast<std::size_t>(width_));
        row_values_.resize(static_cast<std::size_t>(width_));
        row_scales_.resize(static_cast<std::size_t>(width_ / 32));
    }

    void read_into(std::uint64_t relative_offset, std::uint8_t *destination,
                   std::size_t count) const {
        if (relative_offset > nbytes_ || count > nbytes_ - relative_offset) {
            throw std::out_of_range("DeepSeek-V4.1 Engram row range is invalid: " + name_);
        }
        source_->read_range_into(name_, relative_offset, reinterpret_cast<std::byte *>(destination),
                                 count);
    }

    void decode_row(std::int64_t row, std::vector<float> &output) const {
        read_into(engram_detail::kMxHeaderBytes + static_cast<std::uint64_t>(row) * width_,
                  row_values_.data(), row_values_.size());
        read_into(engram_detail::kMxHeaderBytes + value_bytes_ +
                      static_cast<std::uint64_t>(row) * (width_ / 32),
                  row_scales_.data(), row_scales_.size());
        for (int group = 0; group < width_ / 32; ++group) {
            const auto scale = e8m0_[row_scales_[group]];
            if (!std::isfinite(scale)) {
                throw std::runtime_error("DeepSeek-V4.1 Engram row contains an invalid E8M0 scale");
            }
            for (int column = 0; column < 32; ++column) {
                const auto value =
                    e4m3_[row_values_[static_cast<std::size_t>(group * 32 + column)]];
                if (!std::isfinite(value)) {
                    throw std::runtime_error("DeepSeek-V4.1 Engram row contains an E4M3 NaN");
                }
                output[static_cast<std::size_t>(group * 32 + column)] = value * scale;
            }
        }
    }

    const std::vector<float> &row_locked(std::int64_t row) const {
        const auto found = cache_.find(row);
        if (found != cache_.end()) {
            recency_.splice(recency_.begin(), recency_, found->second.recency);
            return found->second.values;
        }
        if (capacity_ == 0) {
            decode_row(row, scratch_);
            return scratch_;
        }
        if (cache_.size() >= capacity_) {
            const auto victim = recency_.back();
            recency_.pop_back();
            cache_.erase(victim);
        }
        recency_.push_front(row);
        CacheEntry entry{std::vector<float>(static_cast<std::size_t>(width_)), recency_.begin()};
        decode_row(row, entry.values);
        return cache_.emplace(row, std::move(entry)).first->second.values;
    }

    const mfq::ModelSource *source_ = nullptr;
    std::string name_;
    std::uint64_t nbytes_ = 0;
    std::int64_t rows_ = 0;
    int width_ = 0;
    std::uint64_t value_bytes_ = 0;
    std::size_t capacity_ = 0;
    std::array<float, 256> e4m3_{};
    std::array<float, 256> e8m0_{};
    mutable std::mutex mutex_;
    mutable std::list<std::int64_t> recency_;
    mutable std::unordered_map<std::int64_t, CacheEntry> cache_;
    mutable std::vector<float> scratch_;
    mutable std::vector<std::uint8_t> row_values_;
    mutable std::vector<std::uint8_t> row_scales_;
};

class Engram {
  public:
    static std::unique_ptr<Engram> load(CudaExecutionContext &execution,
                                        const mfq::ModelSource &model, const EngramConfig &config,
                                        int layer) {
        const auto found =
            std::find(config.engram_layer_ids.begin(), config.engram_layer_ids.end(), layer);
        if (found == config.engram_layer_ids.end()) {
            throw std::invalid_argument("DeepSeek-V4.1 layer has no Engram component");
        }
        const auto hash_layer =
            static_cast<int>(std::distance(config.engram_layer_ids.begin(), found));
        const auto prefix = "model.block." + std::to_string(layer) + ".associative_memory.";
        auto query = load_dense_gpu(execution, model, prefix + "query.weight")
                         .to(mfq_tensor_backend::kFloat32)
                         .contiguous();
        auto key = load_dense_gpu(execution, model, prefix + "key.weight")
                       .to(mfq_tensor_backend::kFloat32)
                       .contiguous();
        if (query.dim() != 2 || query.size(0) != config.hc_mult || query.size(1) != config.hidden ||
            key.sizes() != query.sizes()) {
            throw std::runtime_error("DeepSeek-V4.1 Engram query/key geometry disagrees");
        }
        auto projection = load_quant_linear(execution, model, prefix + "projection.weight");
        const auto hash_columns = (config.engram_max_ngram_size - 1) * config.engram_n_heads;
        if (projection.neuron_len() != hash_columns * config.engram_head_dim ||
            projection.out() != (config.hc_mult + 1) * config.hidden) {
            throw std::runtime_error("DeepSeek-V4.1 Engram projection geometry disagrees");
        }
        return std::unique_ptr<Engram>(new Engram(
            config, hash_layer,
            EngramTable::load(model, prefix + "embedding.weight",
                              config.engram_num_embeddings[static_cast<std::size_t>(hash_layer)],
                              static_cast<int>(config.engram_head_dim),
                              execution.config.deepseek_v41_engram_cache_rows),
            std::move(projection), (query * key).contiguous()));
    }

    EngramTensor forward(CudaExecutionContext &execution, const EngramTensor &hidden,
                         const EngramHashBatch &hashes) const {
        const auto batch = hashes.batch;
        const auto tokens = hashes.tokens;
        if (hidden.dim() != 4 || hidden.size(0) != batch || hidden.size(1) != tokens ||
            hidden.size(2) != config_.hc_mult || hidden.size(3) != config_.hidden ||
            hashes.layers != static_cast<int>(config_.engram_layer_ids.size()) ||
            hashes.columns != (config_.engram_max_ngram_size - 1) * config_.engram_n_heads) {
            throw std::invalid_argument("DeepSeek-V4.1 Engram activation geometry disagrees");
        }
        struct Projection {
            EngramTensor key, value, source;
        };
        return mfq::models::deepseek_v41::engram(
            [&] {
                return table_->gather(hashes.layer_data(hash_layer_), batch, tokens, hashes.columns,
                                      hidden.device());
            },
            [&](EngramTensor embeddings) {
                auto key_value = projection_.forward(execution, embeddings);
                const auto key_width = config_.hc_mult * config_.hidden;
                auto key = key_value.narrow(-1, 0, key_width)
                               .reshape({batch, tokens, config_.hc_mult, config_.hidden})
                               .to(mfq_tensor_backend::kFloat32);
                auto value = key_value.narrow(-1, key_width, config_.hidden)
                                 .reshape({batch, tokens, 1, config_.hidden})
                                 .to(mfq_tensor_backend::kFloat32);
                auto source = hidden.to(mfq_tensor_backend::kFloat32);
                return Projection{key, value, source};
            },
            [&](const Projection &p) {
                const auto &key = p.key;
                const auto &source = p.source;
                auto inverse =
                    mfq_tensor_backend::rsqrt(source.square().mean(-1) + config_.rms_eps) *
                    mfq_tensor_backend::rsqrt(key.square().mean(-1) + config_.rms_eps);
                auto score = (source * key * query_key_weight_).sum(-1) * inverse /
                             std::sqrt(static_cast<double>(config_.hidden));
                auto sign = (score > 0.0).to(mfq_tensor_backend::kFloat32) -
                            (score < 0.0).to(mfq_tensor_backend::kFloat32);
                auto gate = mfq_tensor_backend::sigmoid(sign * score.abs().clamp_min(1e-6).sqrt());
                return gate;
            },
            [&](EngramTensor gate, const Projection &p) {
                const auto &source = p.source;
                const auto &value = p.value;
                return (source + gate.unsqueeze(-1) * value).to(hidden.scalar_type()).contiguous();
            });
    }

  private:
    Engram(EngramConfig config, int hash_layer, std::shared_ptr<EngramTable> table,
           QuantLinear projection, EngramTensor query_key_weight)
        : config_(std::move(config)), hash_layer_(hash_layer), table_(std::move(table)),
          projection_(std::move(projection)), query_key_weight_(std::move(query_key_weight)) {}

    EngramConfig config_;
    int hash_layer_ = -1;
    std::shared_ptr<EngramTable> table_;
    QuantLinear projection_;
    EngramTensor query_key_weight_;
};

} // namespace mfq::cuda::deepseek_v41_runtime
