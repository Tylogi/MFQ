#include "mlx_qsa_kv_offload.h"
#include "mlx_sparse_attention.h"
#include "qwen4_ops.h"
#include "mlx_qwen4_causal_lm.h"
#include "mlx_paged_session_codec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
using namespace mfq::metal;
using namespace mlx::core;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

array values(const Shape& shape, int multiplier, Dtype dtype = float16) {
    int size = 1;
    for (int dimension : shape) size *= dimension;
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) data[i] = (static_cast<int>((i * multiplier) % 251) - 125) / 193.0f;
    return astype(array(data.begin(), shape), dtype);
}

void near(const array& left, const array& right, float tolerance = 0.002f) {
    require(left.shape() == right.shape(), "offload result shape differs from resident reference");
    auto difference = max(abs(astype(left, float32) - astype(right, float32)));
    const float error = difference.item<float>();
    if (!(error < tolerance)) throw std::runtime_error("offload differs from resident reference: max error " +
        std::to_string(error) + ", tolerance " + std::to_string(tolerance));
}

void test_dynamic_store() {
    QsaKvStoreConfig config;
    config.budget_bytes = 32768;
    config.buffer_bytes = 8192;
    config.target_index_bytes = 100000;
    auto store = std::make_shared<QsaKvStore>(config);
    std::vector<QsaKvStore::BlockPtr> index, raw;
    for (int i = 0; i < 4; ++i) index.push_back(store->write(std::vector<std::uint8_t>(4096, i), 2));
    require(store->stats().disk_bytes == 0 && store->stats().hot_bytes == 16384,
        "future indexer reserve prevented full residency below budget");
    for (int i = 0; i < 4; ++i) raw.push_back(store->write(std::vector<std::uint8_t>(4096, i + 10), 0));
    store->flush();
    require(store->stats().hot_bytes <= 24576 && store->stats().disk_bytes == 8192,
        "raw KV did not progressively spill at the shared ceiling");
    auto before = store->stats();
    for (int i = 0; i < 4; ++i) require(store->read(index[i])->at(0) == i, "indexer contents changed");
    require(store->stats().reads == before.reads, "raw KV evicted higher-priority indexer keys");
    for (int i = 4; i < 12; ++i) index.push_back(store->write(std::vector<std::uint8_t>(4096, i), 2));
    store->flush();
    require(store->stats().hot_bytes <= 24576 && store->stats().disk_bytes > 8192,
        "indexer did not spill when it exceeded the budget");
    for (int i = 0; i < 12; ++i) require(store->read(index[i])->at(0) == i, "spilled indexer cannot be read back");
}

void test_snapshot_rollback() {
    QsaKvStoreConfig config;
    config.budget_bytes = 16384;
    config.buffer_bytes = 8192;
    auto store = std::make_shared<QsaKvStore>(config);
    QsaKvSequence sequence(store, 32, 4);
    std::vector<std::uint8_t> original(1000 * 32);
    for (std::size_t i = 0; i < original.size(); ++i) original[i] = i % 251;
    sequence.append(original.data(), 1000);
    auto snapshot = sequence.snapshot();
    auto short_snapshot = snapshot.prefix(997);
    sequence.trim(993);
    std::vector<std::uint8_t> replacement(20 * 32, 255);
    sequence.append(replacement.data(), 20);
    store->flush();
    std::vector<std::uint8_t> read(original.size());
    snapshot.read_rows(0, 1000, read.data());
    require(read == original, "rollback append corrupted immutable snapshot blocks");
    short_snapshot.read_rows(0, 997, read.data());
    require(std::equal(read.begin(), read.begin() + 997 * 32, original.begin()), "unaligned snapshot prefix changed");
    sequence.restore(snapshot);
    require(sequence.position() == 1000, "snapshot restore changed position");
}

void test_store_telemetry() {
    QsaKvStoreConfig config;
    config.budget_bytes = 32768;
    config.buffer_bytes = 8192;
    auto store = std::make_shared<QsaKvStore>(config);
    std::vector<QsaKvStore::BlockPtr> blocks;
    for (int i = 0; i < 8; ++i) blocks.push_back(store->write(std::vector<std::uint8_t>(4096, i), 1));
    store->flush();
    auto before = store->stats();
    require(before.budget_bytes == 32768 && before.disk_bytes == 8192 && before.written_bytes == 8192,
        "SSD telemetry confused resident blocks, pending writes or the byte budget");
    require(before.read_bytes == 0 && before.reads == 0, "creating KV blocks counted as SSD reads");
    require(store->read(blocks.back())->at(0) == 7, "RAM-hit KV contents changed");
    auto hit = store->stats();
    require(hit.hits == before.hits + 1 && hit.read_bytes == 0 && hit.reads == 0,
        "RAM-hit KV telemetry counted file reads");
    require(store->read(blocks.front())->at(0) == 0, "SSD-read KV contents changed");
    store->flush();
    auto read = store->stats();
    require(read.reads == 1 && read.read_bytes == 4096 && read.written_bytes == 12288,
        "KV file traffic is not counted once per completed block read/write");
    blocks.clear();
    auto released = store->stats();
    require(released.hot_bytes == 0 && released.disk_bytes == 0 && released.read_bytes == 4096,
        "releasing KV blocks did not free occupancy or preserve cumulative traffic");
}

void test_shared_resident_guard() {
    QsaKvStoreConfig config;
    config.budget_bytes = 16384;
    config.buffer_bytes = 8192;
    config.reserve = [](std::size_t bytes) {
        require(bytes == 4096, "shared resident guard received the wrong allocation size");
        throw std::runtime_error("shared budget exhausted");
    };
    auto store = std::make_shared<QsaKvStore>(config);
    bool rejected = false;
    try { (void)store->write(std::vector<std::uint8_t>(4096)); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected && store->stats().hot_bytes == 0 && store->stats().disk_bytes == 0,
        "KV allocation bypassed the shared resident guard");
    bool fail = false;
    config.reserve = [&](std::size_t) { if (fail) throw std::runtime_error("shared budget exhausted"); };
    auto guarded_store = std::make_shared<QsaKvStore>(config);
    QsaKvSequence sequence(guarded_store, 1024, 4);
    std::vector<std::uint8_t> original(3072, 43), extra(2048, 71), restored(3072);
    sequence.append(original.data(), 3);
    fail = true;
    try { sequence.seal_tail(); } catch (const std::runtime_error&) {}
    sequence.snapshot().read_rows(0, 3, restored.data());
    require(restored == original, "failed tail allocation destroyed KV rows");
    fail = false;
    sequence.seal_tail();
    fail = true;
    try { sequence.append(extra.data(), 2); } catch (const std::runtime_error&) {}
    sequence.snapshot().read_rows(0, 3, restored.data());
    require(sequence.position() == 3 && restored == original, "failed append changed the committed KV prefix");
}

void test_attention(int queries, int length, int budget, Dtype dtype) {
    QsaKvStoreConfig config;
    config.budget_bytes = 256 * 1024;
    config.buffer_bytes = 64 * 1024;
    auto store = std::make_shared<QsaKvStore>(config);
    MlxQsaKvOffload cache(store, 2, 256, 4096, 4);
    auto key = values({1, 2, length, 256}, 53, dtype);
    auto value = values({1, 2, length, 256}, 71, dtype);
    auto query = values({1, 24, queries, 256}, 37, dtype);
    cache.append(key, value);
    const int offset = length - queries;
    if (length <= budget) {
        near(cache.attention(query, {}, offset, budget), qwen4_dense_gqa_attention(query, key, value, offset));
    } else {
        std::vector<std::int32_t> ids(queries * budget / 4);
        for (int i = 0; i < queries; ++i)
            for (int j = 0; j < budget / 4; ++j) ids[i * budget / 4 + j] = j * 2;
        array blocks(ids.begin(), Shape{1, queries, budget / 4});
        near(cache.attention(query, blocks, offset, budget),
            mlx_sparse_block_gqa_attention(query, key, value, blocks, offset, 4));
    }
    auto snapshot = cache.snapshot();
    cache.trim(length - 3);
    cache.append(slice(key, Shape{0, 0, 0, 0}, Shape{1, 2, 3, 256}),
        slice(value, Shape{0, 0, 0, 0}, Shape{1, 2, 3, 256}));
    auto restored = snapshot.materialize(0, std::min(length, 8));
    near(restored.key, slice(key, Shape{0, 0, 0, 0}, Shape{1, 2, std::min(length, 8), 256}), 0.00001f);
}

void test_indexer(std::size_t budget = 32768, std::size_t buffer = 8192) {
    QsaKvStoreConfig config;
    config.budget_bytes = budget;
    config.buffer_bytes = buffer;
    auto store = std::make_shared<QsaKvStore>(config);
    MlxQsaIndexOffload cache(store, 32, float32, 2);
    auto keys = values({1, 1024, 32}, 43, float32);
    auto query = values({1, 6, 4, 32}, 29);
    cache.append(keys);
    require(budget > 32768 ? store->stats().disk_bytes == 0 : store->stats().disk_bytes > 0,
        "indexer residency does not match its byte budget");
    auto selected = cache.select(query, 4090, 4, 128);
    auto scores = qwen4_qsa_block_scores(query, keys);
    auto positions = reshape(arange(0, 1024, 1, int32), Shape{1, 1, 1024});
    auto visible = reshape(floor_divide(arange(4091, 4097, 1, int32), array(4, int32)), Shape{1, 6, 1});
    scores = where(positions < visible, scores, array(-1e30f));
    auto reference = sort(slice(argpartition(-scores, 31, -1), Shape{0, 0, 0}, Shape{1, 6, 32}), -1);
    require(all(selected == reference).item<bool>(), "streamed indexer top-k differs from full resident scores");
    auto tied = cache.select(zeros(query.shape(), float16), 4090, 4, 128);
    require(all(tied >= array(0, int32)).item<bool>() && all(tied < visible).item<bool>() &&
        all(slice(tied, Shape{0, 0, 1}, tied.shape()) >
            slice(tied, Shape{0, 0, 0}, Shape{1, 6, 31})).item<bool>(),
        "tied streamed top-k selected duplicate or invisible blocks");
}

void test_quantized_attention(double bits, int queries, std::size_t buffer) {
    setenv("MFQ_KV_TURBOQUANT_BITS", std::to_string(bits).c_str(), 1);
    QsaKvStoreConfig config;
    config.buffer_bytes = std::max<std::size_t>(buffer, 16384); config.budget_bytes = config.buffer_bytes * 2;
    auto store = std::make_shared<QsaKvStore>(config);
    MlxQsaKvOffload offloaded(store, 2, 256, 128, 4);
    MlxKvCache resident(1, 2, 128, 256, 4, float16, true);
    auto key = values({1, 2, 39, 256}, 53), value = values({1, 2, 39, 256}, 71);
    auto query = values({1, 24, queries, 256}, 37);
    resident.append_only(key, value); offloaded.append(key, value);
    std::vector<std::int32_t> ids(queries * 4);
    for (int i = 0; i < queries; ++i) for (int j = 0; j < 4; ++j) ids[i * 4 + j] = j * 2;
    const array blocks(ids.begin(), Shape{1, queries, 4});
    near(offloaded.attention(query, blocks, 39 - queries, 16), resident.sparse_attention(query, blocks, 39 - queries, 4, 16));
    auto snapshot = offloaded.snapshot();
    auto physical = snapshot.materialize(0, 4);
    auto reference = resident.snapshot();
    require(snapshot.dtype == uint32 && snapshot.quantization.bits == bits, "offloaded KV is not actually compressed");
    require(snapshot.rows.row_bytes == 2 * 4 * (reference.key.shape(3) + reference.value.shape(3)), "fractional KV row bytes are wrong");
    near(physical.key, slice(reference.key, {0, 0, 0, 0}, {1, 2, 4, reference.key.shape(3)}), .00001f);
    near(physical.value, slice(reference.value, {0, 0, 0, 0}, {1, 2, 4, reference.value.shape(3)}), .00001f);
    offloaded.trim(36); offloaded.restore(snapshot);
    near(offloaded.attention(query, blocks, 39 - queries, 16), resident.sparse_attention(query, blocks, 39 - queries, 4, 16));
    MlxQsaIndexOffload index(store, 128, float32, 2);
    auto index_keys = values({1, 8, 128}, 43, float32);
    index.append(index_keys);
    require(index.snapshot().dtype == float32 && index.snapshot().rows.row_bytes == 512,
        "KV quantization altered Indexer precision");
    near(index.view(0, 8), index_keys, .00001f);
    unsetenv("MFQ_KV_TURBOQUANT_BITS");
}

void test_compressed_attention_reference(double bits, int queries, int dimension, int batch, Dtype dtype,
    int length = 39, int budget = 16) {
    setenv("MFQ_KV_TURBOQUANT_BITS", std::to_string(bits).c_str(), 1);
    const int ratio = 4, kv_heads = 2;
    const int heads = dimension == 256 ? 24 : 6, offset = length - queries;
    MlxKvCache cache(batch, kv_heads, length + 32, dimension, 4, dtype, true);
    cache.append_only(values({batch, kv_heads, length, dimension}, 53, dtype),
        values({batch, kv_heads, length, dimension}, 71, dtype));
    const auto snapshot = cache.snapshot();
    auto key = mlx_kv_decode(snapshot.key, dimension, snapshot.quantization.key_bits(), false, float32);
    auto value = mlx_kv_decode(snapshot.value, dimension, snapshot.quantization.value_bits(), true, float32);
    auto query = values({batch, heads, queries, dimension}, 37, dtype);
    auto fp32_query = astype(query, float32);
    eval(key, value, fp32_query);
    std::vector<std::int32_t> ids(batch * queries * budget / ratio);
    for (int row = 0; row < batch * queries; ++row)
        for (int block = 0; block < budget / ratio; ++block)
            ids[row * budget / ratio + block] = row % 5 == 0 && block == 0 ? -1 :
                row % 3 == 0 && block == 3 ? length / ratio : block * 2;
    const array blocks(ids.begin(), Shape{batch, queries, budget / ratio});
    const auto* q = fp32_query.data<float>();
    const auto* k = key.data<float>();
    const auto* v = value.data<float>();
    std::vector<float> expected(batch * queries * heads * dimension, 0.0f);
    for (int b = 0; b < batch; ++b) for (int t = 0; t < queries; ++t) {
        const int visible = offset + t + 1;
        std::vector<int> selected;
        if (visible <= budget) {
            for (int id = 0; id < visible; ++id) selected.push_back(id);
        } else {
            for (int j = 0; j < budget / ratio; ++j) {
                const int block = ids[(b * queries + t) * budget / ratio + j];
                if (block >= 0 && block < visible / ratio)
                    for (int lane = 0; lane < ratio; ++lane) selected.push_back(block * ratio + lane);
            }
            for (int id = visible / ratio * ratio; id < visible; ++id) selected.push_back(id);
        }
        for (int h = 0; h < heads; ++h) {
            const auto* qr = q + ((b * heads + h) * queries + t) * dimension;
            const int kv_head = h / (heads / kv_heads);
            std::vector<double> weights(selected.size());
            double maximum = -INFINITY;
            for (std::size_t i = 0; i < selected.size(); ++i) {
                const auto* kr = k + ((b * kv_heads + kv_head) * length + selected[i]) * dimension;
                double score = 0;
                for (int d = 0; d < dimension; ++d) score += double(qr[d]) * kr[d];
                weights[i] = score / std::sqrt(double(dimension));
                maximum = std::max(maximum, weights[i]);
            }
            double norm = 0;
            for (auto& weight : weights) { weight = std::exp(weight - maximum); norm += weight; }
            for (int d = 0; d < dimension; ++d) {
                double result = 0;
                for (std::size_t i = 0; i < selected.size(); ++i)
                    result += weights[i] * v[((b * kv_heads + kv_head) * length + selected[i]) * dimension + d];
                expected[((b * queries + t) * heads + h) * dimension + d] = norm > 0 ? result / norm : 0;
            }
        }
    }
    const auto reference = astype(array(expected.begin(), Shape{batch, queries, heads, dimension}), dtype);
    const float tolerance = dtype == float32 ? 0.00002f : dtype == float16 ? 0.0005f : 0.004f;
    near(cache.sparse_attention(query, blocks, offset, ratio, budget), reference, tolerance);
    near(mlx_kv_sparse_attention(query, snapshot.key, snapshot.value, blocks,
        snapshot.quantization, length, offset, ratio, budget), reference, tolerance);
    auto interleave_heads = [](const array& packed) {
        return transpose(contiguous(transpose(packed, {0, 2, 1, 3})), {0, 2, 1, 3});
    };
    near(mlx_kv_sparse_attention(query, interleave_heads(snapshot.key), interleave_heads(snapshot.value), blocks,
        snapshot.quantization, length, offset, ratio, budget), reference, tolerance);
    const auto masked = mlx_kv_sparse_attention(query, snapshot.key, snapshot.value, blocks,
        snapshot.quantization, length, offset, ratio, budget, zeros(Shape{batch, queries, length}, bool_));
    require(all(masked == array(0)).item<bool>(), "fully masked compressed attention is not zero");
    unsetenv("MFQ_KV_TURBOQUANT_BITS");
}

void test_model(const char* path, std::size_t budget) {
    ::unsetenv("MFQ_QSA_KV_BUDGET_BYTES");
    ::unsetenv("MFQ_QSA_KV_TARGET_CONTEXT");
    MfqContainer source(path);
    auto resident = MlxQwen4CausalLm::load(source, 1024);
    std::vector<std::int64_t> prompt(769);
    for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = 1 + i % 30;
    array ids(prompt.begin(), Shape{1, static_cast<int>(prompt.size())});
    auto expected = resident.forward(ids);
    expected.eval();
    const auto live_state = resident.capture_text_session_state(prompt, false);
    for (const auto& layer : live_state.layers) {
        if (!layer.kv) continue;
        require(layer.index_keys->buffer_size() <= 16 * 1024,
            "compact index tail still aliases the complete prefill allocation");
    }
    auto resident_state = resident.capture_text_session_state(prompt);
    for (const auto& layer : resident_state.layers) {
        if (!layer.kv) continue;
        require(layer.index_keys && layer.index_keys->shape(1) <= 4 + kMlxMtpEngineMaximumDraftDepth &&
            layer.index_start + layer.index_keys->shape(1) == layer.position && layer.index_start % 4 == 0,
            "resident QSA retained complete intermediate-key history");
        require(layer.pooled_keys && layer.pooled_keys->shape(1) == layer.position / 4,
            "resident QSA did not eagerly finish block keys");
    }
    ::setenv("MFQ_QSA_KV_BUDGET_BYTES", std::to_string(budget).c_str(), 1);
    ::setenv("MFQ_QSA_KV_TARGET_CONTEXT", "1024", 1);
    auto offloaded = MlxQwen4CausalLm::load(source, 1024);
    auto actual = offloaded.forward(ids);
    actual.eval();
    auto state = offloaded.capture_text_session_state(prompt);
    for (std::size_t layer = 0; layer < state.layers.size(); ++layer) {
        const auto& left = state.layers[layer];
        const auto& right = resident_state.layers[layer];
        if (!left.offloaded_kv) continue;
        require(left.index_keys && left.index_keys->shape(1) <= 4 + kMlxMtpEngineMaximumDraftDepth &&
            left.index_start + left.index_keys->shape(1) == left.position,
            "streaming QSA stored complete intermediate-key history");
        near(*left.index_keys, *right.index_keys, .00001f);
        for (int start = 0; start < state.cache_position; start += 8) {
            const int count = std::min(8, state.cache_position - start);
            auto kv = left.offloaded_kv->materialize(start, count);
            auto expected_key = slice(right.kv->key, Shape{0, 0, start, 0}, Shape{1, 2, start + count, right.kv->key.shape(3)});
            auto expected_value = slice(right.kv->value, Shape{0, 0, start, 0}, Shape{1, 2, start + count, right.kv->value.shape(3)});
            if (kv.quantization.enabled()) {
                near(mlx_kv_decode(kv.key, 256, kv.quantization.key_bits(), false),
                    mlx_kv_decode(expected_key, 256, kv.quantization.key_bits(), false), .003f);
                near(mlx_kv_decode(kv.value, 256, kv.quantization.value_bits(), true),
                    mlx_kv_decode(expected_value, 256, kv.quantization.value_bits(), true), .003f);
            } else {
                near(kv.key, expected_key, .00001f); near(kv.value, expected_value, .00001f);
            }
        }
        MlxQsaIndexOffload index(left.offloaded_pooled_keys->rows.store, 128, float32, 2);
        index.restore(*left.offloaded_pooled_keys);
        float error = 0;
        for (int start = 0; start < index.position(); start += 16) {
            const int count = std::min(16, index.position() - start);
            error = std::max(error, max(abs(index.view(start, count) -
                slice(*right.pooled_keys, Shape{0, start, 0}, Shape{1, start + count, 128}))).item<float>());
        }
        require(error < .00001f, "offloaded pooled index differs from resident keys");
    }
    near(actual, expected, 0.003f);
    auto stats = offloaded.qsa_kv_offload_stats();
    require(stats && (budget > 131072 ? stats->disk_bytes == 0 : stats->disk_bytes > 0) && stats->hot_bytes <= stats->hot_limit,
        "complete model did not enforce its shared KV ceiling");
    std::vector<MlxPagedPayload> encoded;
    for (std::size_t block = 0; block < (prompt.size() + 255) / 256; ++block)
        encoded.push_back(MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(state, 256, block));
    auto restored = MlxPagedSessionCodec<MlxQwen4TextSessionState>::decode(encoded, prompt, 256);
    for (std::size_t layer = 0; layer < state.layers.size(); ++layer) {
        if (!state.layers[layer].offloaded_kv) continue;
        require(restored.layers[layer].index_start == state.layers[layer].index_start,
            "prefix restore changed the compact index tail boundary");
        near(*restored.layers[layer].index_keys, *state.layers[layer].index_keys, .00001f);
    }
    offloaded.clear_cache();
    offloaded.restore_text_session_state(restored);
    array continuation({7, 11, 13}, Shape{1, 3});
    near(offloaded.forward(continuation), resident.forward(continuation), 0.003f);
    offloaded.restore_text_session_state(state);
    resident.restore_text_session_state(resident_state);
    near(offloaded.forward(array({17}, Shape{1, 1})), resident.forward(array({17}, Shape{1, 1})), 0.003f);
    for (int depth : {3, 5}) {
        MlxSamplingParams sampling;
        sampling.mtp_max_draft_tokens = depth;
        std::vector<std::int64_t> left, right;
        resident.generate(prompt, sampling, 16, [&](std::int64_t token) { left.push_back(token); return true; });
        offloaded.generate(prompt, sampling, 16, [&](std::int64_t token) { right.push_back(token); return true; });
        require(!left.empty() && left == right, "complete model MTP output differs after KV offload");
        require(offloaded.last_mtp_stats().cycles > 0, "complete model did not exercise MTP verification");
    }
    ::unsetenv("MFQ_QSA_KV_BUDGET_BYTES");
    ::unsetenv("MFQ_QSA_KV_TARGET_CONTEXT");
    std::cout << "Complete model offload, prefix restore and MTP checks passed\n";
}
}

int main(int argc, char** argv) {
    try {
        if (argc >= 2) { test_model(argv[1], argc >= 3 ? std::stoull(argv[2]) : 131072); return 0; }
        test_dynamic_store();
        test_snapshot_rollback();
        test_store_telemetry();
        test_shared_resident_guard();
        test_indexer();
        test_indexer(8 << 20, 2 << 20);
        for (auto dtype : {float16, bfloat16})
            for (int queries = 1; queries <= 6; ++queries) {
                test_attention(queries, 19, 32, dtype);
                test_attention(queries, 39, 16, dtype);
            }
        test_attention(32, 1027, 16, float16);
        for (double bits : {2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0})
            for (int queries : {1, 2, 3, 4, 5, 6})
                test_quantized_attention(bits, queries, queries % 2 ? 8192 : 131072);
        for (double bits : {2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0}) {
            for (int queries : {1, 2, 3, 4, 5, 6, 32})
                test_compressed_attention_reference(bits, queries, 256, 1, float16);
            test_compressed_attention_reference(bits, 6, 192, 2, float32);
            test_compressed_attention_reference(bits, 6, 192, 2, bfloat16);
            test_compressed_attention_reference(bits, 1, 256, 1, float16, 4099, 2048);
        }
        test_compressed_attention_reference(4, 5, 16, 2, float32, 4099, 2048);
        test_compressed_attention_reference(4, 1, 1, 1, float32, 4099, 2048);
        for (int dimension : {16, 384, 1024}) {
            test_compressed_attention_reference(2.5, 5, dimension, 1, float16);
            test_compressed_attention_reference(6, 5, dimension, 1, float32);
        }
        std::cout << "QSA KV offload tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
