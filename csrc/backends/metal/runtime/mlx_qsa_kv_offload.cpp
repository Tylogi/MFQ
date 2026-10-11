#include "mlx_qsa_kv_offload.h"
#include "qwen4_ops.h"
#include "mlx_resident_budget.h"
#include "mlx_mtp.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace mfq::metal {
namespace {
using mlx::core::array;
using mlx::core::Shape;
std::mutex store_mutex;
std::weak_ptr<QsaKvStore> current_store;

std::size_t environment_size(const char* name, std::size_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    std::size_t consumed = 0;
    const std::string text(value);
    const auto number = std::stoull(text, &consumed);
    if (consumed != text.size() || text[0] == '-' || number > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(std::string("invalid ") + name);
    return static_cast<std::size_t>(number);
}

}

std::shared_ptr<QsaKvStore> mlx_qsa_kv_offload_store() {
    std::lock_guard<std::mutex> guard(store_mutex);
    return current_store.lock();
}
void mlx_set_qsa_kv_offload_store(const std::shared_ptr<QsaKvStore>& store) {
    std::lock_guard<std::mutex> guard(store_mutex);
    current_store = store;
}

std::shared_ptr<QsaKvStore> mlx_configure_qsa_kv_offload(
    int maximum, int layers, int heads, int dimension, int index_width, int ratio) {
    mlx_set_qsa_kv_offload_store({});
    const auto budget = environment_size("MFQ_QSA_KV_BUDGET_BYTES", 0);
    if (!budget) return {};
    const auto target = environment_size("MFQ_QSA_KV_TARGET_CONTEXT", maximum);
    if (!target || target > static_cast<std::size_t>(maximum) || layers <= 0 ||
        heads <= 0 || dimension <= 0 || index_width <= 0 || ratio <= 0)
        throw std::invalid_argument("invalid QSA KV target context or architecture");
    QsaKvStoreConfig config;
    config.budget_bytes = budget;
    config.reserve = [](std::size_t bytes) { MlxResidentBudgetScope::reserve(bytes); };
    config.microblock_rows = ratio;
    config.buffer_bytes = std::min(config.buffer_bytes, budget / 4);
    const auto required_buffer = std::max({std::size_t{4096},
        static_cast<std::size_t>(heads) * dimension * 4 * ratio * 4,
        static_cast<std::size_t>(index_width) * 16 * 4 * 2});
    if (config.buffer_bytes < required_buffer)
        throw std::invalid_argument("QSA KV budget cannot hold one microblock and its I/O staging buffers");
    config.target_index_bytes = static_cast<std::size_t>(layers) *
        (index_width * ((ratio + kMlxMtpEngineMaximumDraftDepth) * std::size_t{2} +
        static_cast<std::size_t>(target) / ratio * 4) + 2);
    const char* directory = std::getenv("MFQ_RUNTIME_PREFIX_CACHE_DIR");
    if (directory) config.directory = directory;
    auto store = std::make_shared<QsaKvStore>(std::move(config));
    mlx_set_qsa_kv_offload_store(store);
    return store;
}

MlxQsaKvOffload::MlxQsaKvOffload(std::shared_ptr<QsaKvStore> store,
    int heads, int dimension, int maximum, int ratio)
    : store_(std::move(store)), quantization_(mlx_kv_quantization()),
      rows_(store_, quantization_.enabled() ? static_cast<std::size_t>(heads) * 4 *
          (mlx_kv_packed_width(dimension, quantization_.key_bits()) + mlx_kv_packed_width(dimension, quantization_.value_bits()))
          : static_cast<std::size_t>(heads) * dimension * 4, ratio, 1),
      heads_(heads), dimension_(dimension), maximum_(maximum), ratio_(ratio) {
    if (quantization_.enabled()) dtype_ = mlx::core::uint32;
}
int MlxQsaKvOffload::position() const noexcept { return static_cast<int>(rows_.position()); }
std::size_t MlxQsaKvOffload::resident_bytes() const noexcept {
    return rows_.view().tail.size();
}

void MlxQsaKvOffload::append(const array& key, const array& value) {
    if (key.ndim() != 4 || key.shape(0) != 1 || key.shape(1) != heads_ || key.shape(3) != dimension_ ||
        value.shape() != key.shape() || value.dtype() != key.dtype() ||
        (key.dtype() != mlx::core::float16 && key.dtype() != mlx::core::bfloat16) ||
        (position() && !quantization_.enabled() && dtype_ != key.dtype()) || key.shape(2) > maximum_ - position())
        throw std::invalid_argument("QSA offloaded KV topology mismatch");
    if (!quantization_.enabled()) dtype_ = key.dtype();
    const int width = heads_ * dimension_;
    const int chunk = static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(256,
        store_->config().buffer_bytes / 8 / (static_cast<std::size_t>(width) * 4))));
    for (int start = 0; start < key.shape(2); start += chunk) {
        const int count = std::min(chunk, key.shape(2) - start);
        auto pack = [&](const array& source, bool value) {
            auto data = mlx::core::transpose(mlx::core::slice(source,
                Shape{0, 0, start, 0}, Shape{1, heads_, start + count, dimension_}), {0, 2, 1, 3});
            if (quantization_.enabled()) data = mlx_kv_encode(data,
                value ? quantization_.value_bits() : quantization_.key_bits(), value);
            return mlx::core::reshape(data, Shape{count, heads_ * data.shape(3)});
        };
        auto packed = mlx::core::contiguous(mlx::core::concatenate({pack(key, false), pack(value, true)}, 1));
        packed.eval();
        rows_.append(packed.data<std::uint8_t>(), count);
    }
    rows_.seal_tail();
}

void MlxQsaKvOffload::append_snapshot(const MlxKvCacheSnapshot& source) {
    if (!source.quantization.enabled()) { append(source.key, source.value); return; }
    if (source.quantization != quantization_ || source.batch != 1 || source.heads != heads_ ||
        source.head_dimension != dimension_ || source.position > maximum_ - position() ||
        source.key.dtype() != dtype_ || source.value.dtype() != dtype_)
        throw std::invalid_argument("quantized QSA snapshot topology mismatch");
    for (int start = 0; start < source.position; start += ratio_) {
        const int count = std::min(ratio_, source.position - start);
        auto pack = [&](const array& data) {
            return mlx::core::reshape(mlx::core::transpose(mlx::core::slice(data, Shape{0, 0, start, 0},
                Shape{1, heads_, start + count, data.shape(3)}), {0, 2, 1, 3}), Shape{count, heads_ * data.shape(3)});
        };
        auto packed = mlx::core::contiguous(mlx::core::concatenate({pack(source.key), pack(source.value)}, 1));
        packed.eval(); rows_.append(packed.data<std::uint8_t>(), count);
    }
    rows_.seal_tail();
}
MlxQsaKvSnapshot MlxQsaKvOffload::snapshot() const {
    return {rows_.snapshot(), heads_, dimension_, maximum_, dtype_, quantization_};
}
void MlxQsaKvOffload::restore(const MlxQsaKvSnapshot& state) {
    if (state.quantization != quantization_ || state.heads != heads_ || state.dimension != dimension_ || state.maximum != maximum_ ||
        state.rows.position > static_cast<std::size_t>(maximum_))
        throw std::invalid_argument("QSA offloaded snapshot topology mismatch");
    rows_.restore(state.rows);
    dtype_ = state.dtype;
}
void MlxQsaKvOffload::trim(int position) {
    if (position < 0) throw std::out_of_range("negative QSA KV trim");
    rows_.trim(position);
}

MlxQsaKvSnapshot MlxQsaKvSnapshot::prefix(int position) const {
    auto result = *this;
    if (position < 0) throw std::out_of_range("negative QSA snapshot prefix");
    result.rows = rows.prefix(position);
    return result;
}

MlxKvCacheSnapshot MlxQsaKvSnapshot::materialize(int start, int count) const {
    if (start < 0 || count <= 0 || static_cast<std::size_t>(count) * rows.row_bytes >
        rows.store->config().buffer_bytes / 2)
        throw std::invalid_argument("QSA KV range exceeds its bounded read buffer");
    const int key_width = quantization.enabled() ? mlx_kv_packed_width(dimension, quantization.key_bits()) : dimension;
    const int value_width = quantization.enabled() ? mlx_kv_packed_width(dimension, quantization.value_bits()) : dimension;
    auto packed = array(mlx::core::allocator::malloc(count * rows.row_bytes),
        Shape{count, heads * (key_width + value_width)}, dtype);
    rows.read_rows(start, count, packed.data<std::uint8_t>());
    auto unpack = [&](bool value) {
        const int width = value ? value_width : key_width, offset = value ? heads * key_width : 0;
        auto source = mlx::core::slice(packed, Shape{0, offset}, Shape{count, offset + heads * width});
        return mlx::core::transpose(mlx::core::reshape(source, Shape{1, count, heads, width}), {0, 2, 1, 3});
    };
    return {1, heads, maximum, dimension, count, count, dtype, unpack(false), unpack(true), quantization};
}

array MlxQsaKvOffload::attention(const array& query, const std::optional<array>& blocks,
    int query_offset, int budget) const {
    if (query.ndim() != 4 || query.shape(0) != 1 || query.shape(3) != dimension_ ||
        query.shape(1) % heads_ || query_offset < 0 || query_offset + query.shape(2) > position() || budget <= 0)
        throw std::invalid_argument("QSA offloaded attention topology mismatch");
    const int tokens = query.shape(2);
    std::optional<array> selected;
    const std::int32_t* ids = nullptr;
    int selected_count = 0;
    if (blocks) {
        selected = mlx::core::contiguous(mlx::core::astype(*blocks, mlx::core::int32));
        if (selected->ndim() != 3 || selected->shape(0) != 1 || selected->shape(1) != tokens ||
            selected->shape(2) > budget / ratio_)
            throw std::invalid_argument("QSA offloaded selection shape mismatch");
        selected->eval();
        ids = selected->data<std::int32_t>();
        selected_count = selected->shape(2);
    }
    const int max_keys = budget + ratio_ - 1;
    const int key_width = quantization_.enabled() ? mlx_kv_packed_width(dimension_, quantization_.key_bits()) : dimension_;
    const int value_width = quantization_.enabled() ? mlx_kv_packed_width(dimension_, quantization_.value_bits()) : dimension_;
    const auto key_bytes = static_cast<std::size_t>(heads_) * key_width * dtype_.size();
    const auto value_bytes = static_cast<std::size_t>(heads_) * value_width * dtype_.size();
    const auto row_bytes = key_bytes + value_bytes;
    if (static_cast<std::size_t>(max_keys) * row_bytes > store_->config().buffer_bytes / 8) {
        const auto& state = rows_.view();
        const int chunk = static_cast<int>(std::max<std::size_t>(1,
            store_->config().buffer_bytes / 8 / row_bytes));
        std::vector<array> outputs;
        for (int token = 0; token < tokens; ++token) {
            const int visible = query_offset + token + 1;
            std::vector<int> positions;
            if (visible <= budget || !ids) {
                if (visible > budget) throw std::runtime_error("missing QSA sparse selection");
                for (int i = 0; i < visible; ++i) positions.push_back(i);
            } else {
                int previous = -1;
                for (int i = 0; i < selected_count; ++i) {
                    const int block = ids[token * selected_count + i];
                    if (block >= 0 && block < visible / ratio_ && block != previous)
                        for (int j = 0; j < ratio_; ++j) positions.push_back(block * ratio_ + j);
                    previous = block;
                }
                for (int i = visible / ratio_ * ratio_; i < visible; ++i) positions.push_back(i);
            }
            if (positions.empty()) throw std::runtime_error("QSA offloaded attention has no visible keys");
            auto q = mlx::core::astype(mlx::core::slice(query, Shape{0, 0, token, 0},
                Shape{1, query.shape(1), token + 1, dimension_}), mlx::core::float32);
            std::optional<array> maximum, norm, weighted;
            for (int begin = 0; begin < static_cast<int>(positions.size()); begin += chunk) {
                const int count = std::min(chunk, static_cast<int>(positions.size()) - begin);
                auto packed = array(mlx::core::allocator::malloc(count * row_bytes),
                    Shape{1, count, heads_ * (key_width + value_width)}, dtype_);
                for (int i = 0; i < count; ++i)
                    state.read_rows(positions[begin + i], 1, packed.data<std::uint8_t>() + i * row_bytes);
                auto expand = [&](bool is_value) {
                    const int width = is_value ? value_width : key_width, offset = is_value ? heads_ * key_width : 0;
                    auto source = mlx::core::slice(packed, Shape{0, 0, offset}, Shape{1, count, offset + heads_ * width});
                    auto value = mlx::core::transpose(mlx::core::reshape(source,
                        Shape{1, count, heads_, width}), {0, 2, 1, 3});
                    if (quantization_.enabled()) value = mlx_kv_decode(value, dimension_,
                        is_value ? quantization_.value_bits() : quantization_.key_bits(), is_value, mlx::core::float32);
                    value = mlx::core::expand_dims(value, 2);
                    value = mlx::core::broadcast_to(value,
                        Shape{1, heads_, query.shape(1) / heads_, count, dimension_});
                    return mlx::core::astype(mlx::core::reshape(value,
                        Shape{1, query.shape(1), count, dimension_}), mlx::core::float32);
                };
                auto scores = mlx::core::matmul(q, mlx::core::swapaxes(expand(false), -1, -2)) /
                    std::sqrt(static_cast<float>(dimension_));
                auto local_max = mlx::core::max(scores, -1, true);
                auto probabilities = mlx::core::exp(scores - local_max);
                auto local_norm = mlx::core::sum(probabilities, -1, true);
                auto local_weighted = mlx::core::matmul(probabilities, expand(true));
                if (maximum) {
                    auto next_max = mlx::core::maximum(*maximum, local_max);
                    auto previous_scale = mlx::core::exp(*maximum - next_max);
                    auto current_scale = mlx::core::exp(local_max - next_max);
                    norm = *norm * previous_scale + local_norm * current_scale;
                    weighted = *weighted * previous_scale + local_weighted * current_scale;
                    maximum = std::move(next_max);
                } else {
                    maximum = std::move(local_max);
                    norm = std::move(local_norm);
                    weighted = std::move(local_weighted);
                }
                mlx::core::eval(*maximum, *norm, *weighted);
            }
            auto output = mlx::core::astype(*weighted / *norm, query.dtype());
            output.eval();
            outputs.push_back(mlx::core::transpose(output, {0, 2, 1, 3}));
        }
        return outputs.size() == 1 ? outputs.front() : mlx::core::concatenate(outputs, 1);
    }
    const int batch_limit = static_cast<int>(std::max<std::size_t>(1,
        store_->config().buffer_bytes / 8 / row_bytes / max_keys));
    const auto& state = rows_.view();
    std::vector<array> results;
    for (int begin = 0; begin < tokens; begin += batch_limit) {
        const int count = std::min(batch_limit, tokens - begin);
        auto keys = array(mlx::core::allocator::malloc(count * max_keys * key_bytes),
            Shape{count, max_keys, heads_, key_width}, dtype_);
        auto values = array(mlx::core::allocator::malloc(count * max_keys * value_bytes),
            Shape{count, max_keys, heads_, value_width}, dtype_);
        auto mask = array(mlx::core::allocator::malloc(count * max_keys), Shape{count, 1, 1, max_keys}, mlx::core::bool_);
        std::memset(keys.data<std::uint8_t>(), 0, keys.nbytes());
        std::memset(values.data<std::uint8_t>(), 0, values.nbytes());
        std::memset(mask.data<std::uint8_t>(), 0, mask.nbytes());
        std::vector<std::uint8_t> row(row_bytes * ratio_);
        for (int i = 0; i < count; ++i) {
            const int visible = query_offset + begin + i + 1;
            int used = 0;
            auto copy = [&](int start, int rows) {
                if (used + rows > max_keys) throw std::runtime_error("QSA selection exceeds its KV buffer");
                state.read_rows(start, rows, row.data());
                for (int j = 0; j < rows; ++j) {
                    const auto offset = static_cast<std::size_t>(i * max_keys + used++);
                    std::memcpy(keys.data<std::uint8_t>() + offset * key_bytes, row.data() + j * row_bytes, key_bytes);
                    std::memcpy(values.data<std::uint8_t>() + offset * value_bytes, row.data() + j * row_bytes + key_bytes, value_bytes);
                    mask.data<std::uint8_t>()[i * max_keys + used - 1] = 1;
                }
            };
            if (visible <= budget || !ids) {
                if (visible > budget) throw std::runtime_error("missing QSA sparse selection");
                for (int start = 0; start < visible; start += ratio_) copy(start, std::min(ratio_, visible - start));
            } else {
                const int complete = visible / ratio_;
                int previous = -1;
                for (int j = 0; j < selected_count; ++j) {
                    const int block = ids[(begin + i) * selected_count + j];
                    if (block >= 0 && block < complete && block != previous) copy(block * ratio_, ratio_);
                    previous = block;
                }
                if (visible % ratio_) copy(complete * ratio_, visible % ratio_);
            }
            if (!used) throw std::runtime_error("QSA offloaded attention has no visible keys");
        }
        auto queries = mlx::core::transpose(mlx::core::slice(query, Shape{0, 0, begin, 0},
            Shape{1, query.shape(1), begin + count, dimension_}), {2, 1, 0, 3});
        if (quantization_.enabled()) {
            auto output = mlx_kv_sparse_attention(queries, mlx::core::transpose(keys, {0, 2, 1, 3}),
                mlx::core::transpose(values, {0, 2, 1, 3}), std::nullopt, quantization_, max_keys,
                max_keys - 1, ratio_, max_keys, mlx::core::reshape(mask, Shape{count, 1, max_keys}));
            output.eval(); results.push_back(mlx::core::transpose(output, {1, 0, 2, 3}));
            continue;
        }
        auto output = scaled_dot_product_attention(queries,
            mlx::core::transpose(keys, {0, 2, 1, 3}), mlx::core::transpose(values, {0, 2, 1, 3}),
            false, 1.0f / std::sqrt(static_cast<float>(dimension_)), mask);
        output.eval();
        results.push_back(mlx::core::transpose(output, {2, 0, 1, 3}));
    }
    return results.size() == 1 ? results.front() : mlx::core::concatenate(results, 1);
}

MlxQsaIndexOffload::MlxQsaIndexOffload(std::shared_ptr<QsaKvStore> store,
    int width, mlx::core::Dtype dtype, int priority)
    : rows_(std::move(store), static_cast<std::size_t>(width) * dtype.size(), 16, priority),
      width_(width), dtype_(dtype) {}

void MlxQsaIndexOffload::append(const array& keys) {
    if (keys.ndim() != 3 || keys.shape(0) != 1 || keys.shape(2) != width_)
        throw std::invalid_argument("offloaded QSA index shape mismatch");
    auto contiguous = mlx::core::contiguous(mlx::core::astype(keys, dtype_));
    contiguous.eval();
    rows_.append(contiguous.data<std::uint8_t>(), keys.shape(1));
    rows_.seal_tail();
}
int MlxQsaIndexOffload::position() const noexcept { return static_cast<int>(rows_.position()); }
void MlxQsaIndexOffload::trim(int position) { rows_.trim(position); }
MlxQsaIndexSnapshot MlxQsaIndexOffload::snapshot() const { return {rows_.snapshot(), width_, dtype_}; }
void MlxQsaIndexOffload::restore(const MlxQsaIndexSnapshot& state) {
    if (state.width != width_ || state.dtype != dtype_) throw std::invalid_argument("offloaded QSA index snapshot mismatch");
    rows_.restore(state.rows);
}
array MlxQsaIndexOffload::view(int start, int count) const {
    if (start < 0 || count <= 0) throw std::invalid_argument("invalid offloaded QSA index range");
    const auto& state = rows_.view();
    if (static_cast<std::size_t>(count) * state.row_bytes > state.store->config().buffer_bytes / 2)
        throw std::invalid_argument("offloaded QSA index range exceeds read buffer");
    auto keys = array(mlx::core::allocator::malloc(count * state.row_bytes), Shape{1, count, width_}, dtype_);
    state.read_rows(start, count, keys.data<std::uint8_t>());
    return keys;
}

array MlxQsaIndexOffload::select(const array& query, int offset, int ratio, int budget) const {
    const int complete = position();
    const int count = std::min(complete, budget / ratio);
    if (count <= 0 || query.ndim() != 4 || query.shape(0) != 1 || query.shape(3) != width_)
        throw std::invalid_argument("invalid offloaded QSA index selection");
    const auto& state = rows_.view();
    const int key_chunk = static_cast<int>(std::max<std::size_t>(1,
        std::min<std::size_t>(complete, state.store->config().buffer_bytes / 8 / state.row_bytes)));
    std::vector<array> results;
    for (int token = 0; token < query.shape(1); token += 8) {
        const int take = std::min(8, query.shape(1) - token);
        auto queries = mlx::core::slice(query, Shape{0, token, 0, 0},
            Shape{1, token + take, query.shape(2), width_});
        const auto visible = mlx::core::reshape(mlx::core::floor_divide(
            mlx::core::arange(offset + token + 1, offset + token + take + 1, 1, mlx::core::int32),
            array(ratio, mlx::core::int32)), Shape{1, take, 1});
        std::optional<array> scores, indices;
        for (int start = 0; start < complete; start += key_chunk) {
            const int length = std::min(key_chunk, complete - start);
            auto ids = mlx::core::broadcast_to(mlx::core::reshape(
                mlx::core::arange(start, start + length, 1, mlx::core::int32), Shape{1, 1, length}),
                Shape{1, take, length});
            auto values = mlx::core::where(ids < visible,
                qwen4_qsa_block_scores(queries, view(start, length)), array(-std::numeric_limits<float>::infinity()));
            if (scores) {
                values = mlx::core::concatenate({*scores, values}, -1);
                ids = mlx::core::concatenate({*indices, ids}, -1);
            }
            const int keep = std::min(count, values.shape(-1));
            if (keep < values.shape(-1)) {
                auto selected = mlx::core::slice(mlx::core::argpartition(-values, keep - 1, -1),
                    Shape{0, 0, 0}, Shape{1, take, keep});
                scores = mlx::core::take_along_axis(values, selected, -1);
                indices = mlx::core::take_along_axis(ids, selected, -1);
            } else { scores = std::move(values); indices = std::move(ids); }
            mlx::core::eval(*scores, *indices);
        }
        auto canonical = mlx::core::broadcast_to(mlx::core::reshape(
            mlx::core::arange(0, count, 1, mlx::core::int32), Shape{1, 1, count}), Shape{1, take, count});
        results.push_back(mlx::core::sort(mlx::core::where(visible <= count, canonical, *indices), -1));
    }
    return results.size() == 1 ? results.front() : mlx::core::concatenate(results, 1);
}

}
