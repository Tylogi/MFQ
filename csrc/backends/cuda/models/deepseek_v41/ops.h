#pragma once
#include "kernels/norm_ops.h"

#include "storage/loaders/weight.h"
#include "storage/mfe_expert_store.h"
#include "csrc/models/deepseek_v41/engram.h"
#include "ops/quant/quant_linear.h"
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include "common/transformer_layer.h"
#include "deepseek_v41/causal_lm.h"

#include "models/common/causal_model_ops.h"
#include "models/deepseek_v4/ops.h"
#include "dspark.h"
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
#include "csrc/backends/cuda/kernels/deepseek_v41.h"
#include "csrc/backends/cuda/kernels/deepseek_v4_attention.h"
#include "csrc/backends/cuda/kernels/deepseek_v4_hc.h"
#include "deepseek_v41/config.h"
#include "models/common/ffn.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::deepseek_v41_runtime {

using Tensor = mfq_tensor_backend::Tensor;
using CommonConfig = mfq::models::deepseek_v41::Config;

struct SharedState {
    CommonConfig config;
    Tensor compressed_kv;
    Tensor index_k;
    Tensor topk;
    Tensor candidate_blocks;
    Tensor previous_pre;
    Tensor attention_meta;
    std::unique_ptr<EngramHashState> engram_hash;
    EngramHashBatch engram_hashes;
    std::vector<Tensor> dspark_target_hiddens;
    bool capture_dspark_targets = false;
    std::int64_t compressed_length = 0;
    std::int64_t ratio = 0;
    std::int64_t cache_position = -1;
    std::int64_t batch = 0;
    std::int64_t tokens = 0;
    std::int64_t next_layer = 0;

    void begin_forward(bool capture, std::size_t target_count) {
        capture_dspark_targets = capture;
        dspark_target_hiddens.clear();
        if (capture)
            dspark_target_hiddens.resize(target_count);
    }

    void capture_dspark_target(std::int64_t layer, const CommonConfig &config,
                               const Tensor &hidden) {
        if (!capture_dspark_targets)
            return;
        const auto found = std::find(config.dspark_target_layer_ids.begin(),
                                     config.dspark_target_layer_ids.end(), layer);
        if (found == config.dspark_target_layer_ids.end())
            return;
        const auto slot =
            static_cast<std::size_t>(std::distance(config.dspark_target_layer_ids.begin(), found));
        MFQ_RUNTIME_CHECK(slot < dspark_target_hiddens.size() && hidden.dim() == 4 &&
                              hidden.size(2) == config.hc_mult && hidden.size(3) == config.hidden,
                          "DeepSeek-V4.1 DSpark target geometry mismatch");
        dspark_target_hiddens[slot] = hidden.mean(2).contiguous();
    }

    Tensor dspark_target_hidden() const {
        MFQ_RUNTIME_CHECK(capture_dspark_targets && !dspark_target_hiddens.empty() &&
                              std::all_of(dspark_target_hiddens.begin(),
                                          dspark_target_hiddens.end(),
                                          [](const Tensor &value) { return value.defined(); }),
                          "DeepSeek-V4.1 DSpark target layers were not fully captured");
        return dspark_target_hiddens.size() == 1
                   ? dspark_target_hiddens.front()
                   : mfq_tensor_backend::cat(dspark_target_hiddens, -1).contiguous();
    }

    void begin_speculative() {
        MFQ_RUNTIME_CHECK(engram_hash, "DeepSeek-V4.1 Engram state is unavailable");
        engram_hash->begin_speculative();
    }

    void commit_speculative() noexcept {
        if (engram_hash)
            engram_hash->commit_speculative();
    }

    void rollback_speculative() {
        MFQ_RUNTIME_CHECK(engram_hash, "DeepSeek-V4.1 Engram state is unavailable");
        engram_hash->rollback_speculative();
    }

    void reset_session(std::int64_t current_batch) {
        compressed_kv = Tensor();
        index_k = Tensor();
        topk = Tensor();
        candidate_blocks = Tensor();
        previous_pre = Tensor();
        engram_hashes = {};
        dspark_target_hiddens.clear();
        capture_dspark_targets = false;
        compressed_length = 0;
        ratio = 0;
        cache_position = -1;
        batch = 0;
        tokens = 0;
        next_layer = 0;
        MFQ_RUNTIME_CHECK(engram_hash && current_batch > 0 &&
                              current_batch <= std::numeric_limits<int>::max(),
                          "DeepSeek-V4.1 Engram state is unavailable");
        engram_hash->reset(static_cast<int>(current_batch));
    }

    void enter_layer(std::int64_t layer, std::int64_t layer_count, std::int64_t current_batch,
                     std::int64_t current_tokens, std::int64_t current_position,
                     const mfq_tensor_backend::Device &device, const Tensor &token_ids) {
        if (layer == 0) {
            compressed_kv = Tensor();
            index_k = Tensor();
            topk = Tensor();
            candidate_blocks = Tensor();
            compressed_length = 0;
            ratio = 0;
            cache_position = current_position;
            batch = current_batch;
            tokens = current_tokens;
            next_layer = 0;
            previous_pre =
                mfq_tensor_backend::zeros({current_batch, current_tokens, 4},
                                          mfq_tensor_backend::TensorOptions().device(device).dtype(
                                              mfq_tensor_backend::kFloat32));
            previous_pre.narrow(2, 0, 1).fill_(1.0);
            MFQ_RUNTIME_CHECK(engram_hash && token_ids.dim() == 2 &&
                                  token_ids.size(0) == current_batch &&
                                  token_ids.size(1) == current_tokens,
                              "DeepSeek-V4.1 Engram token state is unavailable");
            engram_hashes = engram_hash->forward(token_ids, current_position);
        }
        MFQ_RUNTIME_CHECK(layer == next_layer && layer >= 0 && layer < layer_count &&
                              current_batch == batch && current_tokens == tokens &&
                              current_position == cache_position && previous_pre.defined(),
                          "DeepSeek-V4.1 layer execution order changed");
        ++next_layer;
        if (!attention_meta.defined() || attention_meta.device() != device) {
            attention_meta = mfq_tensor_backend::empty(
                {8 * 1024 * 1024}, mfq_tensor_backend::TensorOptions().device(device).dtype(
                                       mfq_tensor_backend::kFloat32));
        }
    }

    Tensor final_collapse(const Tensor &hidden, std::int64_t layer_count) const {
        MFQ_RUNTIME_CHECK(hidden.dim() == 4 && hidden.size(2) == 4 && previous_pre.defined() &&
                              previous_pre.sizes() == hidden.sizes().slice(0, 3) &&
                              next_layer == layer_count,
                          "DeepSeek-V4.1 final Mega-mHC state is incomplete");
        return (previous_pre.to(mfq_tensor_backend::kFloat32).unsqueeze(-1) *
                hidden.to(mfq_tensor_backend::kFloat32))
            .sum(2)
            .to(mfq_tensor_backend::kFloat16)
            .contiguous();
    }
};

struct AttentionState {
    Tensor local_kv;
    Tensor compressed_kv;
    Tensor index_k;
    Tensor partial_kv;
    Tensor partial_score;
    std::int64_t position = 0;
    std::int64_t compressed_length = 0;
    std::int64_t partial_length = 0;
};

inline Tensor weighted_rms(const Tensor &input, const Tensor &weight, double epsilon) {
    const auto shape = input.sizes().vec();
    MFQ_RUNTIME_CHECK(!shape.empty() && weight.numel() == shape.back(),
                      "DeepSeek-V4.1 weighted RMS geometry mismatch");
    return rms_norm_cuda(input.reshape({-1, shape.back()}).to(mfq_tensor_backend::kFloat32), weight,
                         epsilon)
        .reshape(shape)
        .to(mfq_tensor_backend::kFloat16)
        .contiguous();
}

inline Tensor rotate_token_major_tail(Tensor input, const Tensor &positions,
                                      const Dsv4RopeTable &rope, bool inverse = false) {
    MFQ_RUNTIME_CHECK(input.dim() == 3, "DeepSeek-V4.1 RoPE input must be [B,T,D]");
    return dsv4_rotate_rope_tail(input.unsqueeze(1).contiguous(), positions, rope, inverse)
        .squeeze(1)
        .contiguous();
}

inline Tensor empty_topk(std::int64_t batch, std::int64_t tokens,
                         const mfq_tensor_backend::Device &device) {
    return mfq_tensor_backend::empty(
        {batch, tokens, 0},
        mfq_tensor_backend::TensorOptions().device(device).dtype(mfq_tensor_backend::kInt32));
}

inline Tensor normalize_topk(Tensor selected, const Tensor &values, std::int64_t pool_length) {
    selected = selected.to(mfq_tensor_backend::kInt64).contiguous();
    auto valid = (selected >= 0.0) & (selected < static_cast<double>(pool_length));
    valid = valid & mfq_tensor_backend::isneginf(values).logical_not();
    auto masked = mfq_tensor_backend::where(valid, selected, static_cast<double>(pool_length));
    auto ordered = std::get<0>(mfq_tensor_backend::sort(masked, -1, false));
    return mfq_tensor_backend::where(ordered < static_cast<double>(pool_length), ordered, -1.0)
        .to(mfq_tensor_backend::kInt32)
        .contiguous();
}

inline Tensor select_topk(const Tensor &scores, std::int64_t requested, std::int64_t pool_length,
                          const std::optional<Tensor> &source_indices = std::nullopt) {
    const auto width = scores.size(-1);
    const auto count = std::min(requested, width);
    if (count <= 0) {
        return empty_topk(scores.size(0), scores.size(1), scores.device());
    }
    Tensor relative;
    Tensor values;
    if (count == width) {
        relative =
            mfq_tensor_backend::arange(width, scores.options().dtype(mfq_tensor_backend::kInt64))
                .reshape({1, 1, width})
                .expand({scores.size(0), scores.size(1), width})
                .contiguous();
        values = scores;
    } else if (count == 512 && scores.scalar_type() == mfq_tensor_backend::kFloat16) {
        relative =
            dsv4_topk512_cuda(scores.contiguous()).to(mfq_tensor_backend::kInt64).contiguous();
        values = scores.gather(-1, relative);
    } else {
        auto result = mfq_tensor_backend::topk(scores, count, -1, true, false);
        values = std::get<0>(result);
        relative = std::get<1>(result);
    }
    auto selected = source_indices
                        ? source_indices->to(mfq_tensor_backend::kInt64).gather(-1, relative)
                        : relative;
    return normalize_topk(selected, values, pool_length);
}

inline Tensor candidate_blocks_from_scores(const Tensor &scores, std::int64_t ratio,
                                           std::int64_t position, std::int64_t requested_blocks,
                                           std::int64_t block_size) {
    const auto batch = scores.size(0);
    const auto tokens = scores.size(1);
    const auto width = scores.size(2);
    const auto blocks = (width + block_size - 1) / block_size;
    const auto padded = blocks * block_size;
    Tensor padded_scores = scores;
    if (padded != width) {
        padded_scores = mfq_tensor_backend::cat(
            {scores,
             mfq_tensor_backend::full({batch, tokens, padded - width},
                                      -std::numeric_limits<float>::infinity(), scores.options())},
            -1);
    }
    auto block_scores = padded_scores.reshape({batch, tokens, blocks, block_size}).amax(-1);
    auto int_options = scores.options().dtype(mfq_tensor_backend::kInt64);
    auto visible = (mfq_tensor_backend::arange(position + 1, position + tokens + 1, int_options)
                        .reshape({1, tokens, 1}) /
                    static_cast<double>(ratio));
    auto last = mfq_tensor_backend::where(visible > 0.0,
                                          (visible - 1.0) / static_cast<double>(block_size), -1.0);
    auto block_ids = mfq_tensor_backend::arange(blocks, int_options).reshape({1, 1, blocks});
    block_scores = mfq_tensor_backend::where(block_ids == last,
                                             std::numeric_limits<float>::infinity(), block_scores);
    const auto count = std::min(requested_blocks, blocks);
    auto result = mfq_tensor_backend::topk(block_scores, count, -1, true, false);
    auto values = std::get<0>(result);
    auto selected = std::get<1>(result);
    return mfq_tensor_backend::where(mfq_tensor_backend::isneginf(values).logical_not(), selected,
                                     -1.0)
        .to(mfq_tensor_backend::kInt32)
        .contiguous();
}

inline Tensor candidate_positions(const Tensor &blocks, std::int64_t block_size,
                                  std::int64_t pool_length) {
    const auto count = blocks.size(-1);
    auto block64 = blocks.to(mfq_tensor_backend::kInt64);
    auto within =
        mfq_tensor_backend::arange(block_size, blocks.options().dtype(mfq_tensor_backend::kInt64))
            .reshape({1, 1, 1, block_size});
    auto expanded = (block64.unsqueeze(-1) * static_cast<double>(block_size) + within)
                        .reshape({blocks.size(0), blocks.size(1), count * block_size});
    auto valid = (expanded >= 0.0) & (expanded < static_cast<double>(pool_length));
    return mfq_tensor_backend::where(valid, expanded, -1.0)
        .to(mfq_tensor_backend::kInt64)
        .contiguous();
}

struct MhcResult {
    Tensor branch;
    Tensor next_pre;
    Tensor post;
    Tensor combination;
};

struct Block final : ::Block {
    CommonConfig config;
    std::int64_t layer = -1;
    std::int64_t ratio = 0;
    std::int64_t max_context = 0;
    std::shared_ptr<SharedState> shared;
    AttentionState state;
    Tensor current_ids;

    Tensor attention_norm;
    Tensor mlp_norm;
    Tensor query_a_norm;
    Tensor key_value_norm;
    Tensor sinks;
    Tensor attention_mhc_function;
    Tensor attention_mhc_scale;
    Tensor attention_mhc_base;
    Tensor mlp_mhc_function;
    Tensor mlp_mhc_scale;
    Tensor mlp_mhc_base;
    Tensor compressor_norm;
    Tensor index_key_norm;

    QuantLinear query_a;
    QuantLinear query_b;
    QuantLinear key_value;
    QuantLinear output_a;
    QuantLinear output_b;
    QuantLinear compressor_key_value;
    QuantLinear compressor_gate;
    QuantLinear index_query;
    QuantLinear index_key;
    QuantLinear index_score;
    FFN mlp;
    Dsv4RopeTable rope;
    std::unique_ptr<Engram> engram;
    Tensor speculative_local_slots;
    Tensor speculative_slot_ids;
    Tensor speculative_partial_kv;
    Tensor speculative_partial_score;
    std::int64_t speculative_position = -1;
    std::int64_t speculative_compressed_length = 0;
    std::int64_t speculative_partial_length = 0;
    bool speculative_pending = false;

    bool kv_source() const noexcept { return config.is_kv_source(layer); }

    bool index_source() const noexcept { return config.is_index_source(layer); }

    void reset(std::int64_t batch) override {
        const auto half = mfq_tensor_backend::TensorOptions()
                              .device(mfq_tensor_backend::kCUDA)
                              .dtype(mfq_tensor_backend::kFloat16);
        state.local_kv =
            mfq_tensor_backend::zeros({batch, config.sliding_window, config.head_dim}, half);
        state.compressed_kv = Tensor();
        state.index_k = Tensor();
        state.partial_kv = Tensor();
        state.partial_score = Tensor();
        if (kv_source()) {
            const auto capacity = std::max<std::int64_t>(1, max_context / ratio);
            state.compressed_kv =
                mfq_tensor_backend::zeros({batch, capacity, config.head_dim}, half);
            state.index_k =
                mfq_tensor_backend::zeros({batch, capacity, config.index_head_dim}, half);
            if (ratio > 1) {
                const auto fp32 = half.dtype(mfq_tensor_backend::kFloat32);
                state.partial_kv = mfq_tensor_backend::zeros({batch, ratio, config.head_dim}, fp32);
                state.partial_score = mfq_tensor_backend::full(
                    {batch, ratio, config.head_dim}, -std::numeric_limits<float>::infinity(), fp32);
            }
        }
        state.position = 0;
        state.compressed_length = 0;
        state.partial_length = 0;
        speculative_local_slots = Tensor();
        speculative_slot_ids = Tensor();
        speculative_partial_kv = Tensor();
        speculative_partial_score = Tensor();
        speculative_position = -1;
        speculative_pending = false;
        if (layer == 0)
            shared->reset_session(batch);
    }

    bool supports_speculation() const noexcept override { return true; }

    void begin_speculative(std::int64_t draft_tokens) override {
        MFQ_RUNTIME_CHECK(!speculative_pending && draft_tokens > 0 &&
                              draft_tokens <= config.sliding_window && state.local_kv.defined(),
                          "invalid DeepSeek-V4.1 speculative checkpoint");
        speculative_position = state.position;
        speculative_compressed_length = state.compressed_length;
        speculative_partial_length = state.partial_length;
        speculative_slot_ids =
            mfq_tensor_backend::arange(state.position, state.position + draft_tokens,
                                       state.local_kv.options().dtype(mfq_tensor_backend::kInt64))
                .remainder(static_cast<double>(config.sliding_window))
                .to(mfq_tensor_backend::kInt64)
                .contiguous();
        speculative_local_slots = state.local_kv.index_select(1, speculative_slot_ids).clone();
        if (state.partial_kv.defined()) {
            speculative_partial_kv = state.partial_kv.clone();
            speculative_partial_score = state.partial_score.clone();
        }
        speculative_pending = true;
    }

    void commit_speculative() override {
        speculative_pending = false;
        speculative_local_slots = Tensor();
        speculative_slot_ids = Tensor();
        speculative_partial_kv = Tensor();
        speculative_partial_score = Tensor();
        speculative_position = -1;
    }

    void rollback_speculative(std::int64_t position) override {
        MFQ_RUNTIME_CHECK(speculative_pending && position == speculative_position &&
                              speculative_local_slots.defined() && speculative_slot_ids.defined(),
                          "DeepSeek-V4.1 speculative checkpoint is unavailable");
        state.local_kv.index_copy_(1, speculative_slot_ids, speculative_local_slots);
        if (speculative_partial_kv.defined()) {
            state.partial_kv.copy_(speculative_partial_kv);
            state.partial_score.copy_(speculative_partial_score);
        }
        state.position = speculative_position;
        state.compressed_length = speculative_compressed_length;
        state.partial_length = speculative_partial_length;
        commit_speculative();
    }

    void set_token_ids(const Tensor &ids) override { current_ids = ids; }

    MhcResult collapse(CudaProfiler &profiler, const Tensor &residual, const Tensor &previous_pre,
                       const Tensor &function, const Tensor &scale, const Tensor &base,
                       const Tensor &norm, const char *profile) const {
        return profiler.measure(profile, [&]() {
            auto flat = residual.flatten(2).to(mfq_tensor_backend::kFloat32);
            auto inverse = mfq_tensor_backend::rsqrt(flat.square().mean(-1, true) + config.rms_eps);
            auto mixes = mfq_tensor_backend::matmul(flat * inverse, function.transpose(0, 1));
            auto expansion = dsv4_hc_pre_cuda(residual, mixes.contiguous(), scale, base,
                                              config.hc_sinkhorn_iters, config.hc_eps);
            auto next_pre =
                mfq_tensor_backend::sigmoid(mixes.slice(-1, 0, config.hc_mult) * scale.index({0}) +
                                            base.slice(0, 0, config.hc_mult)) +
                config.hc_eps;
            auto branch = (previous_pre.to(mfq_tensor_backend::kFloat32).unsqueeze(-1) *
                           residual.to(mfq_tensor_backend::kFloat32))
                              .sum(2);
            branch = weighted_rms(branch, norm, config.rms_eps);
            return MhcResult{std::move(branch), next_pre.contiguous(), expansion.at(1),
                             expansion.at(2)};
        });
    }

    Tensor expand(CudaProfiler &profiler, const Tensor &branch, const Tensor &residual,
                  const MhcResult &mix, const char *profile) const {
        return profiler.measure(profile, [&]() {
            return dsv4_hc_post_cuda(branch.contiguous(), residual.contiguous(),
                                     mix.post.contiguous(), mix.combination.contiguous());
        });
    }

    Tensor output_projection(CudaExecutionContext &execution, Tensor attention) const {
        return mfq::cuda::deepseek_v4::output_projection(execution, attention, output_a, output_b,
                                                         config.o_groups, true, false);
    }

    std::optional<Tensor> update_compressed_source(CudaExecutionContext &execution,
                                                   const Tensor &input, std::int64_t position) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        Tensor projected_kv, projected_score;
        return mfq::models::deepseek_v41::compress(
            state, tokens, position, ratio,
            [&] {
                return weighted_rms(compressor_key_value.forward(execution, input), compressor_norm,
                                    config.rms_eps);
            },
            [&] {
                projected_kv = compressor_key_value.forward(execution, input)
                                   .to(mfq_tensor_backend::kFloat32)
                                   .contiguous();
                projected_score = compressor_gate.forward(execution, input)
                                      .to(mfq_tensor_backend::kFloat32)
                                      .contiguous();
            },
            [&](int64_t complete, int64_t cutoff) {
                auto kv_groups = projected_kv.narrow(1, 0, cutoff)
                                     .reshape({batch, complete, ratio, config.head_dim});
                auto score_groups = projected_score.narrow(1, 0, cutoff)
                                        .reshape({batch, complete, ratio, config.head_dim});
                auto pooled = (kv_groups * mfq_tensor_backend::softmax(score_groups, 2)).sum(2);
                return weighted_rms(pooled.to(mfq_tensor_backend::kFloat16), compressor_norm,
                                    config.rms_eps);
            },
            [&](int64_t cutoff, int64_t remainder) {
                state.partial_kv.narrow(1, 0, remainder)
                    .copy_(projected_kv.narrow(1, cutoff, remainder));
                state.partial_score.narrow(1, 0, remainder)
                    .copy_(projected_score.narrow(1, cutoff, remainder));
            },
            [&](int64_t token, int64_t slot) {
                state.partial_kv.narrow(1, slot, 1).copy_(projected_kv.narrow(1, token, 1));
                state.partial_score.narrow(1, slot, 1).copy_(projected_score.narrow(1, token, 1));
            },
            [&] {
                auto pooled =
                    (state.partial_kv * mfq_tensor_backend::softmax(state.partial_score, 1))
                        .sum(1, true);
                return weighted_rms(pooled.to(mfq_tensor_backend::kFloat16), compressor_norm,
                                    config.rms_eps);
            },
            [](const auto &emitted) { return mfq_tensor_backend::cat(emitted, 1).contiguous(); });
    }

    void publish_compressed_source(CudaExecutionContext &execution, const Tensor &input,
                                   std::int64_t position) {
        auto latent = update_compressed_source(execution, input, position);
        if (latent) {
            const auto count = latent->size(1);
            const auto begin = state.compressed_length;
            MFQ_RUNTIME_CHECK(begin + count <= state.compressed_kv.size(1),
                              "DeepSeek-V4.1 compressed cache capacity exceeded");
            auto group_positions =
                mfq_tensor_backend::arange(begin * ratio, (begin + count) * ratio, ratio,
                                           latent->options().dtype(mfq_tensor_backend::kInt64));
            auto index_values =
                weighted_rms(index_key.forward(execution, *latent), index_key_norm, config.rms_eps);
            index_values = rotate_token_major_tail(index_values, group_positions, rope);
            index_values = dsv4_fp4_sim_cuda(index_values.contiguous());
            auto compressed = rotate_token_major_tail(*latent, group_positions, rope);
            compressed = deepseek_v41_mxfp4_e4m3_scale_sim_cuda(compressed.contiguous());
            auto rows = mfq_tensor_backend::arange(
                begin, begin + count, latent->options().dtype(mfq_tensor_backend::kInt64));
            state.index_k.index_copy_(1, rows, index_values);
            state.compressed_kv.index_copy_(1, rows, compressed);
            state.compressed_length += count;
        }
        shared->compressed_kv = state.compressed_kv;
        shared->index_k = state.index_k;
        shared->compressed_length = state.compressed_length;
        shared->ratio = ratio;
    }

    Tensor compute_topk(CudaExecutionContext &execution, const Tensor &input, const Tensor &q_rank,
                        const Tensor &positions, std::int64_t position) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        const auto pool_length = shared->compressed_length;
        return mfq::models::deepseek_v41::hierarchical_index(
            ratio, index_source(), layer, config.candidate_source_layer, pool_length,
            [&] { return empty_topk(batch, tokens, input.device()); },
            [&] {
                MFQ_RUNTIME_CHECK(shared->topk.defined(),
                                  "DeepSeek-V4.1 CSA2 consumer has no published index plan");
                return shared->topk;
            },
            [&] {
                auto query =
                    index_query.forward(execution, q_rank)
                        .reshape({batch, tokens, config.index_n_heads, config.index_head_dim})
                        .transpose(1, 2)
                        .contiguous();
                query = dsv4_rotate_rope_tail(query, positions, rope, false)
                            .transpose(1, 2)
                            .contiguous();
                query = dsv4_fp4_sim_cuda(query.to(mfq_tensor_backend::kFloat16).contiguous());
                auto weights = index_score.forward(execution, input)
                                   .reshape({batch, tokens, config.index_n_heads})
                                   .to(mfq_tensor_backend::kFloat16)
                                   .contiguous();
                auto scores = dsv4_indexer_scores_cuda(
                    query, shared->index_k.narrow(1, 0, pool_length).contiguous(), weights,
                    position, ratio);
                return scores;
            },
            [&](Tensor scores) {
                shared->candidate_blocks = candidate_blocks_from_scores(
                    scores, ratio, position, config.candidate_topk_blocks,
                    config.candidate_block_size);
            },
            [&](Tensor scores) {
                MFQ_RUNTIME_CHECK(shared->candidate_blocks.defined(),
                                  "DeepSeek-V4.1 hierarchical indexer has no candidates");
                auto candidates = candidate_positions(shared->candidate_blocks,
                                                      config.candidate_block_size, pool_length);
                auto safe = candidates.clamp(0.0, static_cast<double>(pool_length - 1));
                auto candidate_scores = scores.gather(-1, safe);
                candidate_scores = mfq_tensor_backend::where(
                    candidates >= 0.0, candidate_scores, -std::numeric_limits<float>::infinity());
                return select_topk(candidate_scores, config.index_topk, pool_length, candidates);
            },
            [&](Tensor scores) { return select_topk(scores, config.index_topk, pool_length); },
            [&](Tensor topk) { shared->topk = std::move(topk); });
    }

    Tensor attention_forward(CudaExecutionContext &execution, const Tensor &input,
                             const Tensor &positions, std::int64_t position,
                             const MfqOptional<Tensor> &sequence_lengths) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        return mfq::models::deepseek_v41::attention(
            kv_source(), ratio,
            [&] {
                return weighted_rms(query_a.forward(execution, input), query_a_norm,
                                    config.rms_eps);
            },
            [&](Tensor q_rank) {
                auto query = query_b.forward(execution, q_rank)
                                 .reshape({batch, tokens, config.n_heads, config.head_dim})
                                 .transpose(1, 2)
                                 .contiguous()
                                 .to(mfq_tensor_backend::kFloat32);
                query = dsv4_rotate_rope_tail(query, positions, rope, false);

                return query;
            },
            [&] {
                auto local = weighted_rms(key_value.forward(execution, input), key_value_norm,
                                          config.rms_eps)
                                 .reshape({batch, tokens, config.head_dim});
                local = rotate_token_major_tail(local, positions, rope);
                local = deepseek_v41_mxfp8_e4m3_sim_cuda(local.contiguous());

                return local;
            },
            [&] { publish_compressed_source(execution, input, position); },
            [&] {
                MFQ_RUNTIME_CHECK(shared->compressed_kv.defined() && shared->index_k.defined() &&
                                      shared->ratio == ratio,
                                  "DeepSeek-V4.1 CSA2 consumer ran before its source layer");
            },
            [&](Tensor q_rank) {
                return compute_topk(execution, input, q_rank, positions, position);
            },
            [&](Tensor local) {
                Tensor local_for_attention;
                if (tokens == 1) {
                    auto slot = mfq_tensor_backend::full(
                        {1}, position % config.sliding_window,
                        positions.options().dtype(mfq_tensor_backend::kInt64));
                    state.local_kv.index_copy_(1, slot, local);
                    local_for_attention = state.local_kv;
                } else {
                    const auto history = std::min(position, config.sliding_window);
                    std::vector<Tensor> local_parts;
                    if (history > 0) {
                        auto slots = mfq_tensor_backend::arange(
                                         position - history, position,
                                         positions.options().dtype(mfq_tensor_backend::kInt64))
                                         .remainder(static_cast<double>(config.sliding_window))
                                         .to(mfq_tensor_backend::kInt64)
                                         .contiguous();
                        local_parts.push_back(state.local_kv.index_select(1, slots));
                    }
                    local_parts.push_back(local);
                    local_for_attention =
                        local_parts.size() == 1
                            ? local_parts.front()
                            : mfq_tensor_backend::cat(local_parts, 1).contiguous();
                    const auto recent = std::min(tokens, config.sliding_window);
                    auto recent_slots = mfq_tensor_backend::arange(
                                            position + tokens - recent, position + tokens,
                                            positions.options().dtype(mfq_tensor_backend::kInt64))
                                            .remainder(static_cast<double>(config.sliding_window))
                                            .to(mfq_tensor_backend::kInt64)
                                            .contiguous();
                    state.local_kv.index_copy_(1, recent_slots,
                                               local.narrow(1, tokens - recent, recent));
                }

                const auto pool_length = ratio > 0 ? shared->compressed_length : 0;
                auto cache =
                    pool_length > 0
                        ? mfq_tensor_backend::cat({local_for_attention,
                                                   shared->compressed_kv.narrow(1, 0, pool_length)},
                                                  1)
                              .contiguous()
                        : local_for_attention.contiguous();
                return cache;
            },
            [&](Tensor topk) {
                const auto pool_length = ratio > 0 ? shared->compressed_length : 0;
                std::vector<Tensor> plan;
                if (tokens == 1) {
                    auto lengths = sequence_lengths.has_value()
                                       ? sequence_lengths.value()
                                             .to(input.device(), mfq_tensor_backend::kInt64)
                                             .contiguous()
                                       : mfq_tensor_backend::full(
                                             {batch}, position + 1,
                                             positions.options().dtype(mfq_tensor_backend::kInt64));
                    plan = dsv4_build_decode_plan_cuda(
                        topk, lengths, pool_length, ratio > 0 ? ratio : 1, config.sliding_window);
                } else {
                    MFQ_RUNTIME_CHECK(
                        !sequence_lengths.has_value(),
                        "DeepSeek-V4.1 chunked prefill does not accept per-row lengths");
                    const auto history = std::min(position, config.sliding_window);
                    plan =
                        dsv4_build_prefill_plan_cuda(topk, position, history, pool_length,
                                                     ratio > 0 ? ratio : 1, config.sliding_window);
                }
                return plan;
            },
            [&](Tensor query, Tensor cache, const auto &plan) {
                return attention_dsv4_sparse_cuda(
                    query, cache, plan.at(0), plan.at(1), sinks, shared->attention_meta,
                    1.0 / std::sqrt(static_cast<double>(config.head_dim)));
            },
            [&](Tensor attended) {
                return dsv4_rotate_rope_tail(attended.transpose(1, 2).contiguous(), positions, rope,
                                             true)
                    .transpose(1, 2)
                    .contiguous();
            },
            [&] { state.position += tokens; },
            [&](Tensor attended) { return output_projection(execution, attended); });
    }

    Tensor forward(CudaExecutionContext &execution, Tensor hidden, Tensor positions,
                   std::int64_t cache_position, const MfqOptional<Tensor> &sequence_lengths,
                   const RopeCache &, const MfqOptional<Tensor> &cache_positions = mfq_nullopt,
                   const MfqOptional<Tensor> &attention_mask = mfq_nullopt) override {
        auto &profiler = execution.profiler;
        MFQ_RUNTIME_CHECK(current_ids.defined() && hidden.dim() == 4 &&
                              hidden.size(2) == config.hc_mult && hidden.size(3) == config.hidden &&
                              positions.dim() == 1 && positions.numel() == hidden.size(1) &&
                              !cache_positions.has_value() && !attention_mask.has_value() &&
                              state.position == cache_position,
                          "DeepSeek-V4.1 block input/cache mismatch");
        shared->enter_layer(layer, config.n_layers, hidden.size(0), hidden.size(1), cache_position,
                            hidden.device(), current_ids);

        const auto batch = hidden.size(0), tokens = hidden.size(1);
        return mfq::models::deepseek_v41::decoder_layer(
            std::move(hidden), bool(engram), shared->previous_pre,
            [&](Tensor value) {
                return profiler.measure("deepseek_v41.engram", [&] {
                    return engram->forward(execution, value, shared->engram_hashes);
                });
            },
            [&](const Tensor &value) { shared->capture_dspark_target(layer, config, value); },
            [&](Tensor value, const Tensor &previous_pre, int branch) {
                return collapse(profiler, value, previous_pre,
                                branch == 0 ? attention_mhc_function : mlp_mhc_function,
                                branch == 0 ? attention_mhc_scale : mlp_mhc_scale,
                                branch == 0 ? attention_mhc_base : mlp_mhc_base,
                                branch == 0 ? attention_norm : mlp_norm,
                                branch == 0 ? "deepseek_v41.mhc.attention.collapse"
                                            : "deepseek_v41.mhc.mlp.collapse");
            },
            [&](Tensor value) {
                return attention_forward(execution, value, positions, cache_position,
                                         sequence_lengths);
            },
            [&](Tensor value, Tensor residual, const auto &mix, int branch) {
                return expand(profiler, value, residual, mix,
                              branch == 0 ? "deepseek_v41.mhc.attention.expand"
                                          : "deepseek_v41.mhc.mlp.expand");
            },
            [&](Tensor value) {
                return mlp.forward(execution, value.reshape({-1, config.hidden}), current_ids)
                    .reshape({batch, tokens, config.hidden});
            });
    }
};

FFN load_moe_at(CudaExecutionContext &execution, const mfq::ModelSource &model,
                const CommonConfig &config, const std::string &prefix, std::int64_t layer,
                std::int64_t top_k);

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &model,
                                    std::int64_t layer, const std::shared_ptr<SharedState> &shared);
void validate_load_options(const CudaExecutionContext &execution);
std::shared_ptr<SharedState> load_shared_state(const mfq::ModelSource &source,
                                               const CommonConfig &config);
Tensor finalize_hidden(const std::shared_ptr<SharedState> &state, const Tensor &hidden,
                       CudaProfiler &profiler);

inline int run_self_check() {
    EngramHashState::self_check();
    run_dspark_self_check();
    auto options = mfq_tensor_backend::TensorOptions()
                       .device(mfq_tensor_backend::kCUDA)
                       .dtype(mfq_tensor_backend::kInt32);
    auto blocks =
        mfq_tensor_backend::tensor(std::vector<std::int32_t>{0, 2, -1}, options).reshape({1, 1, 3});
    auto positions = candidate_positions(blocks, 2, 5)
                         .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                         .contiguous();
    const std::vector<std::int64_t> expected{0, 1, 4, -1, -1, -1};
    const auto *values = positions.data_ptr<std::int64_t>();
    MFQ_RUNTIME_CHECK(positions.numel() == static_cast<std::int64_t>(expected.size()) &&
                          std::equal(expected.begin(), expected.end(), values),
                      "DeepSeek-V4.1 candidate hierarchy contract failed");
    std::cout << "deepseek_v41_runtime_check=ok\n";
    return 0;
}

} // namespace mfq::cuda::deepseek_v41_runtime

namespace mfq::cuda {

struct DeepseekV41Model : CausalResources {
    static constexpr CudaFamilyPolicy policy{.predictor = CudaPredictorAdapter::deepseek_v41_dspark};
    static CudaModelPlan plan_components(const mfq::ModelGraph&);
    template <class Backend> using CausalModel = mfq::models::deepseek_v41::CausalLm<Backend>;
    mfq::models::deepseek_v41::Config config;
    std::shared_ptr<deepseek_v41_runtime::SharedState> shared;

    void validate_load_options() const;
    void load_final_state(const mfq::ModelSource &source,
                                  mfq_tensor_backend::Tensor &output_norm);
    void prepare_blocks(const mfq::ModelSource &source);
    std::unique_ptr<Block> load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);

    void begin_forward(bool capture_raw_hidden);
    mfq_tensor_backend::Tensor collapse_hidden(mfq_tensor_backend::Tensor hidden, int64_t batch,
                                               int64_t tokens) const;
    mfq_tensor_backend::Tensor normalize_hidden(mfq_tensor_backend::Tensor hidden,
                                                const mfq_tensor_backend::Tensor &output_norm,
                                                int64_t batch, int64_t tokens) const;
    mfq_tensor_backend::Tensor
    extract_raw_hidden(const mfq_tensor_backend::Tensor &hidden,
                       const mfq_tensor_backend::Tensor &finalized) const;
    void native_begin_speculation();
    void native_commit_speculation();
    void native_rollback_speculation(int64_t keep);
};

extern template struct CudaSessionCodec<DeepseekV41Model>;

} // namespace mfq::cuda

namespace mfq::models {
extern template struct deepseek_v41::CausalLm<cuda::CudaCausalOps<cuda::DeepseekV41Model>>;
} // namespace mfq::models
