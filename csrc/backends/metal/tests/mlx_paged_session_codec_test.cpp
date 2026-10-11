#include "mlx_paged_session_codec.h"

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

mlx::core::array fp16_values(int count, int offset = 0) {
    std::vector<float> values(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        values[static_cast<std::size_t>(index)] =
            static_cast<float>(index + offset) / 16.0f;
    }
    return mlx::core::astype(
        mlx::core::array(
            values.begin(), mlx::core::Shape{1, 2, count / 4, 2}),
        mlx::core::float16);
}

bool byte_equal(const mlx::core::array& left, const mlx::core::array& right) {
    auto evaluated_left = left;
    auto evaluated_right = right;
    evaluated_left.eval();
    evaluated_right.eval();
    return evaluated_left.shape() == evaluated_right.shape() &&
        evaluated_left.dtype() == evaluated_right.dtype() &&
        evaluated_left.nbytes() == evaluated_right.nbytes() &&
        std::memcmp(
            evaluated_left.data<std::uint8_t>(),
            evaluated_right.data<std::uint8_t>(),
            evaluated_left.nbytes()) == 0;
}

void run_codec_benchmark() {
    using namespace mfq::metal;
    constexpr int tokens = 1536;
    constexpr int block_size = 256;
    constexpr int layers = 24;
    constexpr int heads = 8;
    constexpr int head_dim = 64;
    auto key = mlx::core::zeros(
        mlx::core::Shape{1, heads, tokens, head_dim}, mlx::core::float16);
    auto value = mlx::core::ones(
        mlx::core::Shape{1, heads, tokens, head_dim}, mlx::core::float16);
    mlx::core::eval(key, value);
    MlxMiniCPMO45TextSessionState state;
    state.tokens.resize(tokens);
    for (int token = 0; token < tokens; ++token) state.tokens[token] = token;
    state.cache_position = tokens;
    state.cache_batch = 1;
    for (int layer = 0; layer < layers; ++layer) {
        state.layers.push_back(MlxKvCacheSnapshot{
            1,
            heads,
            tokens,
            head_dim,
            tokens,
            tokens,
            mlx::core::float16,
            key,
            value,
        });
        state.bytes += state.layers.back().nbytes();
    }
    const auto encoded =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::encode(
            state, block_size);
    const auto started = std::chrono::steady_clock::now();
    auto decoded =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
            encoded, state.tokens, block_size);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    require(decoded.bytes == state.bytes, "codec benchmark byte count mismatch");
    std::cout << "paged_codec_benchmark bytes=" << state.bytes
              << " blocks=" << encoded.size()
              << " layers=" << layers
              << " elapsed_ms=" << elapsed << '\n';

    auto copied =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
            encoded, state.tokens, block_size);
    const auto copy_started = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<MlxKvCache>> copied_caches;
    for (const auto& layer : copied.layers) {
        auto cache = std::make_unique<MlxKvCache>(
            layer.batch,
            layer.heads,
            layer.maximum_sequence,
            layer.head_dimension,
            layer.position,
            layer.dtype);
        cache->restore_snapshot(layer);
        copied_caches.push_back(std::move(cache));
    }
    const auto copy_elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - copy_started).count();
    const auto adopt_started = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<MlxKvCache>> adopted_caches;
    for (auto& layer : decoded.layers) {
        auto cache = std::make_unique<MlxKvCache>(
            layer.batch,
            layer.heads,
            layer.maximum_sequence,
            layer.head_dimension,
            layer.position,
            layer.dtype);
        cache->restore_snapshot(std::move(layer));
        adopted_caches.push_back(std::move(cache));
    }
    const auto adopt_elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - adopt_started).count();
    std::cout << "paged_codec_restore_benchmark bytes=" << state.bytes
              << " copy_ms=" << copy_elapsed
              << " adopt_ms=" << adopt_elapsed << '\n';
}

} // namespace

int main() {
    using namespace mfq::metal;
    MlxMiniCPMO45TextSessionState mini;
    mini.tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    mini.cache_position = 8;
    mini.cache_batch = 1;
    mini.layers.push_back(MlxKvCacheSnapshot{
        1,
        2,
        32,
        2,
        16,
        8,
        mlx::core::float16,
        fp16_values(32),
        fp16_values(32, 100),
    });
    mini.bytes = mini.layers.front().nbytes();
    const auto encoded =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::encode(mini, 4);
    require(encoded.size() == 2, "MiniCPM block encode count mismatch");
    const auto encoded_second =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::encode_block(
            mini, 4, 1);
    require(*encoded_second == *encoded[1],
            "MiniCPM one-block encoding changed bytes");
    const auto decoded =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
            encoded, mini.tokens, 4);
    require(decoded.tokens == mini.tokens, "MiniCPM decoded tokens mismatch");
    require(decoded.layers.size() == 1, "MiniCPM decoded layers mismatch");
    require(byte_equal(decoded.layers[0].key, mini.layers[0].key),
            "MiniCPM decoded key bytes changed");
    require(byte_equal(decoded.layers[0].value, mini.layers[0].value),
            "MiniCPM decoded value bytes changed");
    auto adopted =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
            encoded, mini.tokens, 4);
    const auto adopted_key = adopted.layers[0].key.buffer().ptr();
    MlxKvCache resumed(1, 2, 32, 2, 1, mlx::core::float16);
    resumed.restore_snapshot(std::move(adopted.layers[0]));
    require(resumed.key_storage().buffer().ptr() == adopted_key &&
                resumed.position() == 8 && resumed.capacity() == 8,
            "decoded KV cache was not adopted directly");
    auto appended = resumed.append(fp16_values(4, 200), fp16_values(4, 300));
    mlx::core::eval(appended.first, appended.second);
    require(resumed.position() == 9 && resumed.capacity() >= 9,
            "adopted KV cache did not grow on append");

    MlxQwen35TextSessionState hybrid;
    hybrid.tokens = mini.tokens;
    hybrid.cache_position = 8;
    hybrid.cache_batch = 1;
    hybrid.layers.emplace_back(mini.layers.front());
    auto convolution = mlx::core::array(
        {1.0f, 2.0f}, mlx::core::Shape{1, 2});
    auto recurrent = mlx::core::array(
        {3.0f, 4.0f}, mlx::core::Shape{1, 2});
    hybrid.layers.emplace_back(MlxQwen35LinearAttentionCacheSnapshot{
        convolution,
        recurrent,
        8,
        1,
    });
    const auto hybrid_encoded =
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode(hybrid, 4);
    require(MlxPagedSessionCodec<MlxQwen35TextSessionState>::available,
            "hybrid SSD codec was not enabled");
    require(
        !MlxPagedSessionCodec<MlxQwen35TextSessionState>::has_exact_boundary(
            hybrid_encoded[0]),
        "non-final hybrid block captured a future recurrent state");
    require(
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::has_exact_boundary(
            hybrid_encoded[1]),
        "stable hybrid boundary did not retain recurrent state");
    require(hybrid_encoded[0]->size() < hybrid_encoded[1]->size(),
            "non-boundary hybrid block retained recurrent payload bytes");
    require(
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::decodable_blocks(
            hybrid_encoded) == 2,
        "hybrid codec did not select the newest exact boundary");
    require(
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::decodable_blocks(
            {hybrid_encoded[0]}) == 0,
        "hybrid codec accepted a block without recurrent state");
    const auto hybrid_second =
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode_block(
            hybrid, 4, 1);
    require(*hybrid_second == *hybrid_encoded[1],
            "hybrid one-block encoding changed bytes");
    const auto hybrid_decoded =
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::decode(
            hybrid_encoded, hybrid.tokens, 4);
    require(hybrid_decoded.layers.size() == 2,
            "hybrid decoded layer count mismatch");
    const auto& decoded_recurrent =
        std::get<MlxQwen35LinearAttentionCacheSnapshot>(
            hybrid_decoded.layers[1]);
    require(byte_equal(decoded_recurrent.convolution_state, convolution),
            "recurrent convolution bytes changed");
    require(byte_equal(decoded_recurrent.recurrent_state, recurrent),
            "recurrent state bytes changed");
    require(decoded_recurrent.position == 8,
            "recurrent cache position mismatch");

    auto unaligned_hybrid = hybrid;
    unaligned_hybrid.tokens.resize(6);
    unaligned_hybrid.cache_position = 6;
    std::get<MlxKvCacheSnapshot>(unaligned_hybrid.layers[0]).position = 6;
    std::get<MlxQwen35LinearAttentionCacheSnapshot>(
        unaligned_hybrid.layers[1]).position = 6;
    const auto unaligned_encoded =
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode(
            unaligned_hybrid, 4);
    require(
        !MlxPagedSessionCodec<MlxQwen35TextSessionState>::has_exact_boundary(
            unaligned_encoded[0]),
        "unaligned final state was mislabeled as an exact block boundary");

    MlxQwen35TextSessionState short_hybrid;
    short_hybrid.tokens.assign(hybrid.tokens.begin(), hybrid.tokens.begin() + 4);
    short_hybrid.cache_position = 4;
    short_hybrid.cache_batch = 1;
    auto short_kv = mini.layers.front();
    short_kv.position = 4;
    short_hybrid.layers.emplace_back(std::move(short_kv));
    auto short_convolution = mlx::core::array(
        {11.0f, 12.0f}, mlx::core::Shape{1, 2});
    auto short_recurrent = mlx::core::array(
        {13.0f, 14.0f}, mlx::core::Shape{1, 2});
    short_hybrid.layers.emplace_back(MlxQwen35LinearAttentionCacheSnapshot{
        short_convolution,
        short_recurrent,
        4,
        1,
    });
    const auto short_encoded =
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode(
            short_hybrid, 4);

    auto long_hybrid = hybrid;
    long_hybrid.tokens.insert(
        long_hybrid.tokens.end(), {9, 10, 11, 12});
    long_hybrid.cache_position = 12;
    std::get<MlxKvCacheSnapshot>(long_hybrid.layers[0]).position = 12;
    std::get<MlxQwen35LinearAttentionCacheSnapshot>(
        long_hybrid.layers[1]).position = 12;
    const auto non_final_second =
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode_block(
            long_hybrid, 4, 1);
    const std::vector<MlxPagedPayload> partial_chain{
        short_encoded[0], non_final_second};
    require(
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::decodable_blocks(
            partial_chain) == 1,
        "hybrid codec did not back off to the previous exact boundary");
    const auto short_decoded =
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::decode(
            {partial_chain[0]}, short_hybrid.tokens, 4);
    const auto& restored_short_recurrent =
        std::get<MlxQwen35LinearAttentionCacheSnapshot>(
            short_decoded.layers[1]);
    require(
        byte_equal(
            restored_short_recurrent.recurrent_state,
            short_recurrent),
        "hybrid fallback restored the wrong recurrent checkpoint");

    const auto hybrid_tail = MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode_block(
        unaligned_hybrid, 4, 1);
    const auto tail_decoded = MlxPagedSessionCodec<MlxQwen35TextSessionState>::decode(
        {unaligned_encoded[0], hybrid_tail}, unaligned_hybrid.tokens, 4);
    require(tail_decoded.cache_position == 6 &&
        byte_equal(std::get<MlxQwen35LinearAttentionCacheSnapshot>(tail_decoded.layers[1]).recurrent_state,
            recurrent), "hybrid tail did not retain its exact state");
    auto short_tail = unaligned_hybrid;
    short_tail.tokens.resize(2);
    short_tail.cache_position = 2;
    std::get<MlxKvCacheSnapshot>(short_tail.layers[0]).position = 2;
    std::get<MlxQwen35LinearAttentionCacheSnapshot>(short_tail.layers[1]).position = 2;
    short_tail.last_hidden = mlx::core::ones({1, 1, 4}, mlx::core::float16);
    auto head = mini.layers.front();
    head.position = 1;
    short_tail.mtp_layers.push_back(head);
    const auto short_payload = MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode_block(short_tail, 4, 0);
    require(MlxPagedSessionCodec<MlxQwen35TextSessionState>::has_mtp(short_payload) &&
        !MlxPagedSessionCodec<MlxQwen35TextSessionState>::has_mtp(hybrid_tail),
        "hybrid codec did not distinguish predictor-ready checkpoints");
    const auto short_restored = MlxPagedSessionCodec<MlxQwen35TextSessionState>::decode(
        {short_payload}, short_tail.tokens, 4);
    require(short_restored.cache_position == 2 && short_restored.mtp_layers.size() == 1 &&
        short_restored.mtp_layers[0].position == 1 &&
        byte_equal(*short_restored.last_hidden, *short_tail.last_hidden),
        "short hybrid MTP tail did not round trip");

    MlxQwen4TextSessionState flash;
    flash.tokens = unaligned_hybrid.tokens;
    flash.cache_position = 6;
    flash.cache_batch = 1;
    MlxQwen4LayerCacheSnapshot qsa;
    qsa.position = 6;
    qsa.batch = 1;
    qsa.index_ratio = 2;
    qsa.kv = std::get<MlxKvCacheSnapshot>(unaligned_hybrid.layers[0]);
    qsa.index_keys = mlx::core::ones({1, 6, 3}, mlx::core::float16);
    qsa.pooled_keys = mlx::core::ones({1, 3, 3}, mlx::core::float32);
    qsa.ple_convolution = convolution;
    qsa.ple_context = {3, 4};
    flash.layers.push_back(qsa);
    MlxQwen4LayerCacheSnapshot gdn;
    gdn.position = 6;
    gdn.batch = 1;
    gdn.convolution = convolution;
    gdn.recurrent = recurrent;
    gdn.ple_convolution = convolution;
    gdn.ple_context = {5, 6};
    flash.layers.push_back(gdn);
    flash.last_hidden = mlx::core::ones({1, 1, 2, 4}, mlx::core::float16);
    auto flash_head = qsa;
    flash_head.position = 5;
    flash_head.kv->position = 5;
    flash_head.index_keys = mlx::core::ones({1, 5, 3}, mlx::core::float16);
    flash_head.pooled_keys = mlx::core::ones({1, 2, 3}, mlx::core::float32);
    flash_head.ple_convolution.reset();
    flash_head.ple_context.clear();
    flash.mtp_layers.push_back(flash_head);
    const auto flash_first = MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(flash, 4, 0);
    const auto flash_last = MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(flash, 4, 1);
    require(MlxPagedSessionCodec<MlxQwen4TextSessionState>::has_mtp(flash_last) &&
        !MlxPagedSessionCodec<MlxQwen4TextSessionState>::has_mtp(flash_first),
        "Flash-Next codec did not distinguish predictor-ready checkpoints");
    require(!MlxPagedSessionCodec<MlxQwen4TextSessionState>::has_exact_boundary(flash_first) &&
        MlxPagedSessionCodec<MlxQwen4TextSessionState>::has_exact_boundary(flash_last),
        "Flash-Next accepted a non-checkpoint block");
    const auto flash_restored = MlxPagedSessionCodec<MlxQwen4TextSessionState>::decode(
        {flash_first, flash_last}, flash.tokens, 4);
    require(byte_equal(*flash_restored.layers[0].index_keys, *qsa.index_keys) &&
        byte_equal(*flash_restored.layers[0].pooled_keys, *qsa.pooled_keys) &&
        byte_equal(*flash_restored.layers[1].recurrent, recurrent) &&
        flash_restored.layers[0].ple_context == qsa.ple_context &&
        flash_restored.layers[1].ple_context == gdn.ple_context &&
        flash_restored.mtp_layers.size() == 1 && flash_restored.mtp_layers[0].position == 5,
        "Flash-Next QSA/GDN/PLE/MTP checkpoint did not round trip");
    QsaKvStoreConfig offload_config;
    offload_config.budget_bytes = 16384;
    offload_config.buffer_bytes = 8192;
    auto offload_store = std::make_shared<QsaKvStore>(offload_config);
    mlx_set_qsa_kv_offload_store(offload_store);
    const auto offloaded = MlxPagedSessionCodec<MlxQwen4TextSessionState>::decode(
        {flash_first, flash_last}, flash.tokens, 4);
    require(!offloaded.layers[0].kv && offloaded.layers[0].index_keys &&
        offloaded.layers[0].offloaded_kv &&
        offloaded.layers[0].offloaded_pooled_keys && offloaded.mtp_layers[0].offloaded_kv,
        "prefix restore expanded offloaded QSA or MTP caches into full MLX arrays");
    const auto offloaded_first = MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(offloaded, 4, 0);
    const auto offloaded_last = MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(offloaded, 4, 1);
    mlx_set_qsa_kv_offload_store({});
    const auto offloaded_roundtrip = MlxPagedSessionCodec<MlxQwen4TextSessionState>::decode(
        {offloaded_first, offloaded_last}, flash.tokens, 4);
    require(byte_equal(offloaded_roundtrip.layers[0].kv->key, flash_restored.layers[0].kv->key) &&
        byte_equal(offloaded_roundtrip.layers[0].kv->value, flash_restored.layers[0].kv->value) &&
        byte_equal(*offloaded_roundtrip.layers[0].index_keys, *flash_restored.layers[0].index_keys) &&
        byte_equal(*offloaded_roundtrip.layers[0].pooled_keys, *flash_restored.layers[0].pooled_keys) &&
        byte_equal(offloaded_roundtrip.mtp_layers[0].kv->key, flash_restored.mtp_layers[0].kv->key),
        "offloaded cache serialization changed persisted KV/indexer/MTP bytes");
    for (double bits : {2.5, 4.0, 8.0}) {
        auto quantized = flash;
        for (auto* list : {&quantized.layers, &quantized.mtp_layers}) for (auto& layer : *list) if (layer.kv) {
            auto& kv = *layer.kv;
            kv.quantization = {bits}; kv.dtype = mlx::core::uint32;
            kv.key = mlx_kv_encode(mlx::core::slice(kv.key, {0, 0, 0, 0}, {kv.batch, kv.heads, kv.position, kv.head_dimension}),
                kv.quantization.key_bits(), false);
            kv.value = mlx_kv_encode(mlx::core::slice(kv.value, {0, 0, 0, 0}, {kv.batch, kv.heads, kv.position, kv.head_dimension}),
                kv.quantization.value_bits(), true);
            mlx::core::eval(kv.key, kv.value);
        }
        const std::vector<MlxPagedPayload> payloads{
            MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(quantized, 4, 0),
            MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(quantized, 4, 1)};
        const auto decoded = MlxPagedSessionCodec<MlxQwen4TextSessionState>::decode(payloads, quantized.tokens, 4);
        require(decoded.layers[0].kv->quantization.bits == bits &&
            byte_equal(decoded.layers[0].kv->key, quantized.layers[0].kv->key) &&
            byte_equal(decoded.layers[0].kv->value, quantized.layers[0].kv->value) &&
            byte_equal(*decoded.layers[0].pooled_keys, *quantized.layers[0].pooled_keys),
            "quantized prefix changed packed KV or FP32 Indexer bytes");
        mlx_set_qsa_kv_offload_store(offload_store);
        const auto streamed = MlxPagedSessionCodec<MlxQwen4TextSessionState>::decode(payloads, quantized.tokens, 4);
        require(streamed.layers[0].offloaded_kv->quantization.bits == bits, "streamed prefix lost quantization metadata");
        const std::vector<MlxPagedPayload> streamed_payloads{
            MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(streamed, 4, 0),
            MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(streamed, 4, 1)};
        mlx_set_qsa_kv_offload_store({});
        const auto restored = MlxPagedSessionCodec<MlxQwen4TextSessionState>::decode(streamed_payloads, quantized.tokens, 4);
        require(byte_equal(restored.layers[0].kv->key, quantized.layers[0].kv->key) &&
            byte_equal(restored.layers[0].kv->value, quantized.layers[0].kv->value) &&
            byte_equal(*restored.layers[0].pooled_keys, *quantized.layers[0].pooled_keys) &&
            restored.mtp_layers[0].kv->quantization.bits == bits,
            "RAM/SSD prefix round trip requantized or expanded KV");
    }
    flash.layers.pop_back();
    const auto qsa_nonfinal = MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(flash, 4, 0);
    require(!MlxPagedSessionCodec<MlxQwen4TextSessionState>::has_exact_boundary(qsa_nonfinal),
        "QSA-only prefix accepted a missing pooled/PLE checkpoint");

    if (const auto* enabled = std::getenv("MFQ_PAGED_CODEC_BENCHMARK");
        enabled != nullptr && enabled[0] == '1') {
        run_codec_benchmark();
    }

    std::cout << "MFQ MLX paged session codec tests passed\n";
    return 0;
}
