// Checks for the shared fused Q/K norm + rotary + KV-store operator.
//
// The operator is compared against a double-precision reference over a geometry
// sweep (head counts, head width, partial rotary, norm offset, MRoPE axes,
// linear/ring/paged stores, BF16 and FP16). A FullBlock case then runs the
// fused and the separate spans over the same weights and compares both the
// output and the KV cache contents. An optional benchmark reports the profiler
// sections and launch counts for decode and prefill.

#include "cuda_execution.h"
#include "core/full_block.h"
#include "core/kv_cache.h"
#include "mfq_cuda_qk_rope_kv.h"
#include "mfq_tensor_backend.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace tb = mfq_tensor_backend;
using tb::Tensor;
using mfq_tensor_backend::indexing::Slice;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Round-to-nearest-even BF16, matching __float2bfloat16_rn on the device.
float round_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    const std::uint32_t half = (bits >> 16) & 1u;
    bits += 0xffu + half;
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float round_storage(float value, tb::ScalarType dtype) {
    return dtype == tb::kBFloat16 ? round_bf16(value) : __half2float(__float2half_rn(value));
}

std::vector<float> pattern(std::size_t count, float amplitude, float salt) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index)
        values[index] = amplitude * std::sin(float(index) * 0.137f + salt) +
                        0.2f * amplitude * std::cos(float(index) * 0.051f - salt);
    return values;
}

Tensor upload(const std::vector<float>& values, std::initializer_list<std::int64_t> shape,
              tb::ScalarType dtype) {
    return tb::tensor(values).reshape(shape).to(dtype).to(tb::kCUDA).contiguous();
}

std::vector<float> host_values(const Tensor& value) {
    auto host = value.contiguous().to(tb::kCPU).to(tb::kFloat32);
    return std::vector<float>(host.data_ptr<float>(), host.data_ptr<float>() + host.numel());
}

void set_environment(const char* name, bool enabled) {
#ifdef _WIN32
    _putenv_s(name, enabled ? "1" : "0");
#else
    setenv(name, enabled ? "1" : "0", 1);
#endif
}

// Operator geometry and destination of one fused case.
struct FusionCase {
    std::string name;
    std::int64_t heads = 4;
    std::int64_t kv_heads = 2;
    std::int64_t head_dim = 64;
    std::int64_t rotary_dim = 64;
    std::int64_t batches = 1;
    std::int64_t tokens = 3;
    double eps = 1e-6;
    double weight_offset = 1.0;
    tb::ScalarType dtype = tb::kFloat16;
    bool normalize = true;
    bool mrope = false;
    bool mrope_per_batch = false;
    bool per_batch_positions = false;
    bool dense_key_value = false;
    int store = 1; // 0 none, 1 linear, 2 ring, 3 paged
    std::int64_t capacity = 8;
    std::int64_t page_size = 2;
    std::int64_t pages_per_chunk = 4;
    std::int64_t chunks = 2;
    std::int64_t table_length = 16;
    std::vector<std::int64_t> sections;
    std::vector<std::int64_t> write_positions;
};

// Exact tensors of one fused case, kept on the host for the reference.
struct CaseTensors {
    Tensor query, key, value;
    Tensor query_weight, key_weight;
    Tensor positions, cosine, sine, sections;
    std::vector<float> query_source, key_source, value_source;
    std::vector<float> query_weight_values, key_weight_values;
    std::vector<float> cosine_values, sine_values;
    std::vector<std::int64_t> position_values;
    std::int64_t position_axes = 1;
    std::int64_t position_batches = 1;
};

std::int64_t table_index(const FusionCase& spec, const CaseTensors& tensors, std::int64_t column,
                         std::int64_t batch, std::int64_t token) {
    std::int64_t axis = 0;
    if (!spec.sections.empty()) {
        const auto first = spec.sections[0];
        const auto second = spec.sections[0] + spec.sections[1];
        axis = column < first ? 0 : (column < second ? 1 : 2);
        if (axis >= tensors.position_axes) axis = 0;
    }
    const auto batch_index = tensors.position_batches == 1 ? 0 : batch;
    const auto index = (axis * tensors.position_batches + batch_index) * spec.tokens + token;
    return std::clamp<std::int64_t>(tensors.position_values[static_cast<std::size_t>(index)], 0,
                                    spec.table_length - 1);
}

Tensor make_positions(const FusionCase& spec, std::vector<std::int64_t>& values,
                      std::int64_t& axes, std::int64_t& batches) {
    const auto tokens = spec.tokens;
    const auto batches_in = spec.batches;
    if (spec.mrope) {
        require(!spec.sections.empty(), "MRoPE cases need sections");
        axes = 3;
        batches = spec.mrope_per_batch ? batches_in : 1;
        values.resize(static_cast<std::size_t>(axes * batches * tokens));
        for (std::int64_t axis = 0; axis < axes; ++axis)
            for (std::int64_t batch = 0; batch < batches; ++batch)
                for (std::int64_t token = 0; token < tokens; ++token) {
                    // Distinct per axis and per batch: a permutation mix-up in
                    // the axis or batch index cannot stay unnoticed.
                    const auto index = (axis * batches + batch) * tokens + token;
                    values[static_cast<std::size_t>(index)] =
                        1 + axis * 3 + batch * 2 + token;
                }
        return tb::tensor(values)
            .reshape({axes, batches, tokens})
            .to(tb::kInt64)
            .to(tb::kCUDA)
            .contiguous();
    }
    axes = 1;
    batches = spec.per_batch_positions ? batches_in : 1;
    values.resize(static_cast<std::size_t>(batches * tokens));
    for (std::int64_t batch = 0; batch < batches; ++batch)
        for (std::int64_t token = 0; token < tokens; ++token) {
            const auto index = batch * tokens + token;
            values[static_cast<std::size_t>(index)] = token + (batches > 1 ? batch : 0);
        }
    return tb::tensor(values)
        .reshape({batches, tokens})
        .to(tb::kInt64)
        .to(tb::kCUDA)
        .contiguous();
}

CaseTensors make_case_tensors(const FusionCase& spec) {
    CaseTensors tensors;
    const auto batches = spec.batches, tokens = spec.tokens, heads = spec.heads,
               kv_heads = spec.kv_heads, width = spec.head_dim;
    tensors.query_source = pattern(
        static_cast<std::size_t>(batches * tokens * heads * width), 0.4f, 1.0f);
    tensors.key_source = pattern(
        static_cast<std::size_t>(batches * tokens * kv_heads * width), 0.5f, 2.0f);
    tensors.value_source = pattern(
        static_cast<std::size_t>(batches * tokens * kv_heads * width), 0.6f, 3.0f);
    tensors.query_weight_values = pattern(static_cast<std::size_t>(width), 0.15f, 4.0f);
    tensors.key_weight_values = pattern(static_cast<std::size_t>(width), 0.2f, 5.0f);
    tensors.query = upload(tensors.query_source, {batches, tokens, heads, width}, spec.dtype);
    tensors.key = upload(tensors.key_source, {batches, tokens, kv_heads, width}, spec.dtype);
    tensors.value = upload(tensors.value_source, {batches, tokens, kv_heads, width}, spec.dtype);
    tensors.query_source = host_values(tensors.query);
    tensors.key_source = host_values(tensors.key);
    tensors.value_source = host_values(tensors.value);
    if (spec.normalize) {
        tensors.query_weight =
            upload(tensors.query_weight_values, {width}, tb::kFloat32);
        tensors.key_weight = upload(tensors.key_weight_values, {width}, tb::kFloat32);
        tensors.query_weight_values = host_values(tensors.query_weight);
        tensors.key_weight_values = host_values(tensors.key_weight);
    }
    tensors.positions = make_positions(spec, tensors.position_values, tensors.position_axes,
                                       tensors.position_batches);
    // Rotate-half table with the reachable positions; the basis does not matter
    // because the reference reads the same stored cos/sin values.
    const auto pairs = spec.rotary_dim / 2;
    tensors.cosine_values.resize(static_cast<std::size_t>(spec.table_length * pairs));
    tensors.sine_values.resize(static_cast<std::size_t>(spec.table_length * pairs));
    for (std::int64_t position = 0; position < spec.table_length; ++position)
        for (std::int64_t pair = 0; pair < pairs; ++pair) {
            const double frequency = std::pow(10000.0, -2.0 * double(pair) / double(spec.rotary_dim));
            const double angle = double(position) * frequency;
            const auto index = static_cast<std::size_t>(position * pairs + pair);
            tensors.cosine_values[index] = float(std::cos(angle));
            tensors.sine_values[index] = float(std::sin(angle));
        }
    tensors.cosine = upload(tensors.cosine_values, {spec.table_length, pairs}, tb::kFloat32);
    tensors.sine = upload(tensors.sine_values, {spec.table_length, pairs}, tb::kFloat32);
    if (!spec.sections.empty())
        tensors.sections = tb::tensor(spec.sections).to(tb::kCPU).to(tb::kInt64).contiguous();
    return tensors;
}

// Reference: normalize (+offset), rotate-half with the stored tables, then
// round to the storage dtype exactly where the operator stores.
std::vector<double> reference_row(const FusionCase& spec, const CaseTensors& tensors,
                                  const std::vector<float>& source, std::int64_t heads,
                                  std::int64_t batch, std::int64_t head, std::int64_t token,
                                  const std::vector<float>* weight) {
    const auto width = spec.head_dim;
    const auto pairs = spec.rotary_dim / 2;
    const bool bf16 = spec.dtype == tb::kBFloat16;
    const auto base = ((batch * spec.tokens + token) * heads + head) * width;
    std::vector<double> normalized(static_cast<std::size_t>(width));
    double squares = 0.0;
    for (std::int64_t column = 0; column < width; ++column) {
        const double value = double(source[static_cast<std::size_t>(base + column)]);
        squares += value * value;
    }
    const double inverse = 1.0 / std::sqrt(squares / double(width) + spec.eps);
    for (std::int64_t column = 0; column < width; ++column) {
        const double value = double(source[static_cast<std::size_t>(base + column)]);
        if (weight == nullptr) {
            normalized[static_cast<std::size_t>(column)] = value;
            continue;
        }
        if (!bf16) {
            normalized[static_cast<std::size_t>(column)] =
                value * inverse * (double((*weight)[static_cast<std::size_t>(column)]) + spec.weight_offset);
            continue;
        }
        const double scaled = double(round_bf16(float(value * inverse)));
        double scale = double(round_bf16((*weight)[static_cast<std::size_t>(column)]));
        if (spec.weight_offset != 0.0) scale = double(round_bf16(float(scale + spec.weight_offset)));
        normalized[static_cast<std::size_t>(column)] = double(round_bf16(float(scale * scaled)));
    }
    std::vector<double> row = normalized;
    for (std::int64_t pair = 0; pair < pairs; ++pair) {
        const auto index = table_index(spec, tensors, pair, batch, token);
        double cosine = double(tensors.cosine_values[static_cast<std::size_t>(index * pairs + pair)]);
        double sine = double(tensors.sine_values[static_cast<std::size_t>(index * pairs + pair)]);
        if (bf16) {
            cosine = double(round_bf16(float(cosine)));
            sine = double(round_bf16(float(sine)));
        }
        const double first = normalized[static_cast<std::size_t>(pair)];
        const double second = normalized[static_cast<std::size_t>(pair + pairs)];
        if (bf16) {
            row[static_cast<std::size_t>(pair)] = double(round_bf16(
                float(round_bf16(float(first * cosine)) - round_bf16(float(second * sine)))));
            row[static_cast<std::size_t>(pair + pairs)] = double(round_bf16(
                float(round_bf16(float(second * cosine)) + round_bf16(float(first * sine)))));
        } else {
            row[static_cast<std::size_t>(pair)] = first * cosine - second * sine;
            row[static_cast<std::size_t>(pair + pairs)] = second * cosine + first * sine;
        }
    }
    for (auto& value : row) value = double(round_storage(float(value), spec.dtype));
    return row;
}

// Cache destination of one fused case.
struct StoreFixture {
    mfq::cuda::QkKvStore store;
    KVCache cache;
    Tensor k_chunks, v_chunks, chunk_k, chunk_v, chunk_k_second, chunk_v_second;
    std::int64_t logical_pages = 0;
};

StoreFixture make_store(const FusionCase& spec) {
    StoreFixture fixture;
    const auto batches = spec.batches, kv_heads = spec.kv_heads, width = spec.head_dim;
    const auto options = tb::TensorOptions().device(tb::kCUDA).dtype(spec.dtype);
    if (spec.store == 0) return fixture;
    auto positions = tb::tensor(spec.write_positions)
                         .reshape({batches, spec.tokens})
                         .to(tb::kInt64)
                         .to(tb::kCUDA)
                         .contiguous();
    fixture.store.positions = positions;
    if (spec.store == 3) {
        fixture.logical_pages = 4;
        fixture.chunk_k = tb::full({spec.pages_per_chunk, kv_heads, spec.page_size, width}, -7.5,
                                   options);
        fixture.chunk_v = tb::full({spec.pages_per_chunk, kv_heads, spec.page_size, width}, -7.5,
                                   options);
        fixture.chunk_k_second = tb::full(
            {spec.pages_per_chunk, kv_heads, spec.page_size, width}, -7.5, options);
        fixture.chunk_v_second = tb::full(
            {spec.pages_per_chunk, kv_heads, spec.page_size, width}, -7.5, options);
        std::vector<std::int64_t> k_pointers{
            static_cast<std::int64_t>(reinterpret_cast<std::intptr_t>(fixture.chunk_k.data_ptr())),
            static_cast<std::int64_t>(
                reinterpret_cast<std::intptr_t>(fixture.chunk_k_second.data_ptr()))};
        std::vector<std::int64_t> v_pointers{
            static_cast<std::int64_t>(reinterpret_cast<std::intptr_t>(fixture.chunk_v.data_ptr())),
            static_cast<std::int64_t>(
                reinterpret_cast<std::intptr_t>(fixture.chunk_v_second.data_ptr()))};
        fixture.k_chunks = tb::tensor(k_pointers).to(tb::kInt64).to(tb::kCUDA).contiguous();
        fixture.v_chunks = tb::tensor(v_pointers).to(tb::kInt64).to(tb::kCUDA).contiguous();
        // Each batch owns a disjoint physical page range and rotates its own
        // mapping, so the reference never sees two writers sharing a page. The
        // second batch starts at the chunk boundary.
        require(batches * fixture.logical_pages <= spec.chunks * spec.pages_per_chunk,
                "paged fixture geometry");
        std::vector<std::int32_t> pages;
        for (std::int64_t batch = 0; batch < batches; ++batch)
            for (std::int64_t page = 0; page < fixture.logical_pages; ++page)
                pages.push_back(std::int32_t(batch * fixture.logical_pages +
                                             (page + batch) % fixture.logical_pages));
        auto page_table = tb::tensor(pages)
                              .reshape({batches, fixture.logical_pages})
                              .to(tb::kInt32)
                              .to(tb::kCUDA)
                              .contiguous();
        fixture.cache = KVCache::paged_view(fixture.k_chunks, fixture.v_chunks, page_table, batches,
                                            kv_heads, width, spec.page_size, spec.pages_per_chunk,
                                            spec.dtype);
        fixture.store.mode = mfq::cuda::QkKvStoreMode::Paged;
        fixture.store.page_table = page_table;
        fixture.store.key_chunks = fixture.k_chunks;
        fixture.store.value_chunks = fixture.v_chunks;
        fixture.store.page_size = spec.page_size;
        fixture.store.pages_per_chunk = spec.pages_per_chunk;
        return fixture;
    }
    const bool ring = spec.store == 2;
    fixture.cache = KVCache(batches, kv_heads, spec.capacity, width, ring,
                            tb::Device(tb::kCUDA), spec.dtype);
    fixture.cache.k.fill_(-7.5);
    fixture.cache.v.fill_(-7.5);
    fixture.store.mode = ring ? mfq::cuda::QkKvStoreMode::Ring : mfq::cuda::QkKvStoreMode::Linear;
    fixture.store.key_cache = fixture.cache.k;
    fixture.store.value_cache = fixture.cache.v;
    return fixture;
}

// Reference tolerance. BF16 rounds every rotary product to eight significant
// bits, so a one-ulp difference between the RMS inverse of the reference (double
// accumulation) and of the kernel (FP32 tree) can flip a product rounding; a
// rotated value that cancels two products then moves by more than its own
// rounding step. FP16 keeps FP32 intermediates and needs no such allowance.
double reference_tolerance(tb::ScalarType dtype) {
    return dtype == tb::kBFloat16 ? 4.0e-2 : 2.0e-3;
}

// Fused against the separate stages: both round at the same points, only the
// reduction order differs.
double integration_tolerance(tb::ScalarType dtype) {
    return dtype == tb::kBFloat16 ? 2.0e-2 : 2.0e-3;
}

void require_close(const std::vector<float>& actual, const std::vector<double>& expected,
                   double tolerance, const std::string& label) {
    require(actual.size() == expected.size(), label + " size mismatch");
    for (std::size_t index = 0; index < expected.size(); ++index)
        require(std::isfinite(actual[index]) &&
                    std::abs(double(actual[index]) - expected[index]) <=
                        tolerance + tolerance * std::abs(expected[index]),
                label + " differs at " + std::to_string(index) + " actual " +
                    std::to_string(actual[index]) + " expected " +
                    std::to_string(expected[index]));
}

void check_operator_case(const FusionCase& spec) {
    auto tensors = make_case_tensors(spec);
    auto store_fixture = make_store(spec);
    mfq::cuda::QkRopeKvArgs args;
    args.query = tensors.query;
    args.key = tensors.key;
    args.value = tensors.value;
    args.query_weight = tensors.query_weight;
    args.key_weight = tensors.key_weight;
    args.eps = spec.eps;
    args.weight_offset = spec.weight_offset;
    args.positions = tensors.positions;
    args.cosine = tensors.cosine;
    args.sine = tensors.sine;
    args.sections = tensors.sections;
    args.rotary_dim = spec.rotary_dim;
    args.store = store_fixture.store;
    args.dense_key_value = spec.dense_key_value;
    const auto fused = mfq::cuda::qk_norm_rope_kv_cuda(args);
    const auto tolerance = reference_tolerance(spec.dtype);

    std::vector<double> expected_query(
        static_cast<std::size_t>(spec.batches * spec.heads * spec.tokens * spec.head_dim));
    std::vector<double> expected_key(
        static_cast<std::size_t>(spec.batches * spec.kv_heads * spec.tokens * spec.head_dim));
    std::vector<double> expected_value = expected_key;
    for (std::int64_t batch = 0; batch < spec.batches; ++batch)
        for (std::int64_t head = 0; head < spec.heads; ++head)
            for (std::int64_t token = 0; token < spec.tokens; ++token) {
                const auto row = reference_row(
                    spec, tensors, tensors.query_source, spec.heads, batch, head, token,
                    spec.normalize ? &tensors.query_weight_values : nullptr);
                const auto base = ((batch * spec.heads + head) * spec.tokens + token) * spec.head_dim;
                for (std::int64_t column = 0; column < spec.head_dim; ++column)
                    expected_query[static_cast<std::size_t>(base + column)] =
                        row[static_cast<std::size_t>(column)];
            }
    for (std::int64_t batch = 0; batch < spec.batches; ++batch)
        for (std::int64_t head = 0; head < spec.kv_heads; ++head)
            for (std::int64_t token = 0; token < spec.tokens; ++token) {
                const auto base = ((batch * spec.kv_heads + head) * spec.tokens + token) *
                                  spec.head_dim;
                const auto key_row = reference_row(
                    spec, tensors, tensors.key_source, spec.kv_heads, batch, head, token,
                    spec.normalize ? &tensors.key_weight_values : nullptr);
                for (std::int64_t column = 0; column < spec.head_dim; ++column)
                    expected_key[static_cast<std::size_t>(base + column)] =
                        key_row[static_cast<std::size_t>(column)];
                const auto source = ((batch * spec.tokens + token) * spec.kv_heads + head) *
                                    spec.head_dim;
                for (std::int64_t column = 0; column < spec.head_dim; ++column)
                    expected_value[static_cast<std::size_t>(base + column)] = double(
                        round_storage(tensors.value_source[static_cast<std::size_t>(source + column)],
                                      spec.dtype));
            }
    require_close(host_values(fused.query), expected_query, tolerance, spec.name + " query");
    if (spec.dense_key_value) {
        require_close(host_values(fused.key), expected_key, tolerance, spec.name + " key");
        require_close(host_values(fused.value), expected_value, tolerance, spec.name + " value");
    } else {
        require(!fused.key.defined() && !fused.value.defined(),
                spec.name + " must not materialize dense key/value rows");
    }

    // Expected cache contents; untouched slots keep the fill value.
    const auto slot_of = [&](std::int64_t batch, std::int64_t token) {
        return spec.write_positions[static_cast<std::size_t>(batch * spec.tokens + token)];
    };
    if (spec.store == 1 || spec.store == 2) {
        const auto elements = static_cast<std::size_t>(
            spec.batches * spec.kv_heads * spec.capacity * spec.head_dim);
        std::vector<double> expected_k(elements, -7.5), expected_v(elements, -7.5);
        for (std::int64_t batch = 0; batch < spec.batches; ++batch)
            for (std::int64_t head = 0; head < spec.kv_heads; ++head)
                for (std::int64_t token = 0; token < spec.tokens; ++token) {
                    auto slot = slot_of(batch, token);
                    if (spec.store == 2) slot %= spec.capacity;
                    if (slot < 0 || slot >= spec.capacity) continue;
                    const auto destination =
                        ((batch * spec.kv_heads + head) * spec.capacity + slot) * spec.head_dim;
                    const auto source =
                        ((batch * spec.kv_heads + head) * spec.tokens + token) * spec.head_dim;
                    for (std::int64_t column = 0; column < spec.head_dim; ++column) {
                        expected_k[static_cast<std::size_t>(destination + column)] =
                            expected_key[static_cast<std::size_t>(source + column)];
                        expected_v[static_cast<std::size_t>(destination + column)] =
                            expected_value[static_cast<std::size_t>(source + column)];
                    }
                }
        require_close(host_values(store_fixture.cache.k), expected_k, tolerance,
                      spec.name + " key cache");
        require_close(host_values(store_fixture.cache.v), expected_v, tolerance,
                      spec.name + " value cache");
    }
    if (spec.store == 3) {
        const auto chunk_elements = static_cast<std::size_t>(
            spec.pages_per_chunk * spec.kv_heads * spec.page_size * spec.head_dim);
        std::vector<double> chunks_k(2 * chunk_elements, -7.5), chunks_v(2 * chunk_elements, -7.5);
        auto page_table = store_fixture.store.page_table;
        const auto pages = host_values(page_table.to(tb::kFloat32));
        for (std::int64_t batch = 0; batch < spec.batches; ++batch)
            for (std::int64_t head = 0; head < spec.kv_heads; ++head)
                for (std::int64_t token = 0; token < spec.tokens; ++token) {
                    const auto position = slot_of(batch, token);
                    const auto logical = position / spec.page_size;
                    require(logical < store_fixture.logical_pages, "paged fixture position range");
                    const auto physical = std::int64_t(
                        pages[static_cast<std::size_t>(batch * store_fixture.logical_pages + logical)]);
                    const auto chunk = physical / spec.pages_per_chunk;
                    const auto local_page = physical % spec.pages_per_chunk;
                    const auto destination =
                        static_cast<std::size_t>(chunk) * chunk_elements +
                        static_cast<std::size_t>((local_page * spec.kv_heads + head) *
                                                     spec.page_size +
                                                 position % spec.page_size) *
                            spec.head_dim;
                    const auto source =
                        ((batch * spec.kv_heads + head) * spec.tokens + token) * spec.head_dim;
                    for (std::int64_t column = 0; column < spec.head_dim; ++column) {
                        chunks_k[destination + static_cast<std::size_t>(column)] =
                            expected_key[static_cast<std::size_t>(source + column)];
                        chunks_v[destination + static_cast<std::size_t>(column)] =
                            expected_value[static_cast<std::size_t>(source + column)];
                    }
                }
        std::vector<float> actual_k = host_values(store_fixture.chunk_k);
        std::vector<float> actual_v = host_values(store_fixture.chunk_v);
        const auto second_k = host_values(store_fixture.chunk_k_second);
        const auto second_v = host_values(store_fixture.chunk_v_second);
        actual_k.insert(actual_k.end(), second_k.begin(), second_k.end());
        actual_v.insert(actual_v.end(), second_v.begin(), second_v.end());
        require_close(actual_k, chunks_k, tolerance, spec.name + " paged key cache");
        require_close(actual_v, chunks_v, tolerance, spec.name + " paged value cache");
    }
}

std::vector<FusionCase> operator_cases() {
    std::vector<FusionCase> cases;
    // Decode-shaped linear store, FP16 and BF16.
    for (auto dtype : {tb::kFloat16, tb::kBFloat16}) {
        FusionCase decode;
        decode.name = std::string("decode-line") + (dtype == tb::kBFloat16 ? "-bf16" : "-f16");
        decode.dtype = dtype;
        decode.batches = 2;
        decode.tokens = 1;
        decode.heads = 8;
        decode.kv_heads = 2;
        decode.head_dim = 64;
        decode.rotary_dim = 64;
        decode.capacity = 6;
        decode.write_positions = {3, 5};
        decode.dense_key_value = false;
        cases.push_back(decode);
    }
    // Prefill with several tokens, partial rotary and no norm offset.
    {
        FusionCase prefill;
        prefill.name = "prefill-partial-rotary-offset0";
        prefill.dtype = tb::kBFloat16;
        prefill.batches = 2;
        prefill.tokens = 4;
        prefill.heads = 6;
        prefill.kv_heads = 3;
        prefill.head_dim = 64;
        prefill.rotary_dim = 32;
        prefill.weight_offset = 0.0;
        prefill.capacity = 8;
        prefill.write_positions = {2, 3, 4, 5, 0, 1, 2, 3};
        prefill.dense_key_value = true;
        cases.push_back(prefill);
    }
    // Exactly filling the cache and a narrower head width.
    {
        FusionCase fill;
        fill.name = "prefill-exact-fill";
        fill.dtype = tb::kFloat16;
        fill.batches = 1;
        fill.tokens = 4;
        fill.heads = 3;
        fill.kv_heads = 1;
        fill.head_dim = 32;
        fill.rotary_dim = 32;
        fill.capacity = 4;
        fill.weight_offset = 1.0;
        fill.write_positions = {0, 1, 2, 3};
        fill.dense_key_value = true;
        cases.push_back(fill);
    }
    // Ring cache with wraparound slots and a wider head.
    {
        FusionCase ring;
        ring.name = "decode-ring-wrap";
        ring.dtype = tb::kBFloat16;
        ring.batches = 2;
        ring.tokens = 2;
        ring.heads = 4;
        ring.kv_heads = 4;
        ring.head_dim = 128;
        ring.rotary_dim = 64;
        ring.capacity = 4;
        ring.store = 2;
        ring.write_positions = {2, 3, 4, 5};
        ring.dense_key_value = false;
        cases.push_back(ring);
    }
    // Paged store crossing a page and a chunk boundary.
    {
        FusionCase paged;
        paged.name = "prefill-paged-cross-chunk";
        paged.dtype = tb::kBFloat16;
        paged.batches = 2;
        paged.tokens = 4;
        paged.heads = 4;
        paged.kv_heads = 2;
        paged.head_dim = 64;
        paged.rotary_dim = 64;
        paged.store = 3;
        paged.page_size = 2;
        paged.write_positions = {0, 1, 2, 3, 4, 5, 6, 7};
        paged.dense_key_value = true;
        cases.push_back(paged);
    }
    // Paged decode without dense rows.
    {
        FusionCase paged;
        paged.name = "decode-paged";
        paged.dtype = tb::kFloat16;
        paged.batches = 2;
        paged.tokens = 1;
        paged.heads = 4;
        paged.kv_heads = 2;
        paged.head_dim = 64;
        paged.rotary_dim = 64;
        paged.store = 3;
        paged.write_positions = {5, 6};
        paged.dense_key_value = false;
        cases.push_back(paged);
    }
    // MRoPE with three distinct axes over a batch, plus clamping past the table.
    {
        FusionCase mrope;
        mrope.name = "mrope-axes-batched";
        mrope.dtype = tb::kBFloat16;
        mrope.batches = 3;
        mrope.tokens = 3;
        mrope.heads = 4;
        mrope.kv_heads = 2;
        mrope.head_dim = 64;
        mrope.rotary_dim = 48;
        mrope.mrope = true;
        mrope.mrope_per_batch = true;
        mrope.sections = {6, 12, 6};
        mrope.table_length = 12;
        mrope.capacity = 8;
        mrope.write_positions = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        mrope.dense_key_value = true;
        cases.push_back(mrope);
    }
    // MRoPE shared across the batch (the single-image layout).
    {
        FusionCase mrope;
        mrope.name = "mrope-axes-shared";
        mrope.dtype = tb::kFloat16;
        mrope.batches = 1;
        mrope.tokens = 2;
        mrope.heads = 4;
        mrope.kv_heads = 1;
        mrope.head_dim = 64;
        mrope.rotary_dim = 24;
        mrope.mrope = true;
        mrope.sections = {4, 4, 4};
        mrope.table_length = 8;
        mrope.capacity = 4;
        mrope.write_positions = {1, 2};
        cases.push_back(mrope);
    }
    // Per-batch positions over three batches.
    {
        FusionCase batched;
        batched.name = "decode-per-batch-positions";
        batched.dtype = tb::kFloat16;
        batched.batches = 3;
        batched.tokens = 1;
        batched.heads = 4;
        batched.kv_heads = 2;
        batched.head_dim = 64;
        batched.rotary_dim = 64;
        batched.per_batch_positions = true;
        batched.capacity = 8;
        batched.write_positions = {1, 2, 3};
        cases.push_back(batched);
    }
    // Rotaries only, no norm weights, no store.
    {
        FusionCase plain;
        plain.name = "rotary-only";
        plain.dtype = tb::kFloat16;
        plain.normalize = false;
        plain.batches = 2;
        plain.tokens = 2;
        plain.heads = 4;
        plain.kv_heads = 4;
        plain.head_dim = 64;
        plain.rotary_dim = 64;
        plain.store = 0;
        plain.dense_key_value = true;
        cases.push_back(plain);
    }
    // Full rotary over a 256-wide head with four query heads per KV head.
    {
        FusionCase wide;
        wide.name = "prefill-wide-head";
        wide.dtype = tb::kBFloat16;
        wide.batches = 1;
        wide.tokens = 2;
        wide.heads = 8;
        wide.kv_heads = 2;
        wide.head_dim = 256;
        wide.rotary_dim = 256;
        wide.table_length = 16;
        wide.capacity = 4;
        wide.write_positions = {0, 1};
        wide.dense_key_value = true;
        cases.push_back(wide);
    }
    return cases;
}

// FullBlock fixture with separate Q/K norm weights and a rotary table.
struct BlockGeometry {
    std::int64_t hidden = 16;
    std::int64_t heads = 4;
    std::int64_t kv_heads = 2;
    std::int64_t head_dim = 64;
    std::int64_t rotary_dim = 64;
    double eps = 1e-6;
    double weight_offset = 1.0;
    std::int64_t max_positions = 32;
    tb::ScalarType dtype = tb::kFloat16;
};

QuantLinear dense_linear(Tensor weight) {
    QuantLinear linear;
    linear.kind = QuantLinearKind::Dense;
    linear.dense = std::move(weight);
    return linear;
}

std::unique_ptr<FullBlock> make_block(const BlockGeometry& geometry) {
    auto block = std::make_unique<FullBlock>();
    const auto width = geometry.heads * geometry.head_dim;
    const auto kv_width = geometry.kv_heads * geometry.head_dim;
    const auto dtype = geometry.dtype;
    block->qkv.layers = {
        dense_linear(upload(pattern(static_cast<std::size_t>(width * geometry.hidden), 0.061f, 1.0f),
                            {width, geometry.hidden}, dtype)),
        dense_linear(upload(pattern(static_cast<std::size_t>(kv_width * geometry.hidden), 0.071f, 2.0f),
                            {kv_width, geometry.hidden}, dtype)),
        dense_linear(upload(pattern(static_cast<std::size_t>(kv_width * geometry.hidden), 0.053f, 3.0f),
                            {kv_width, geometry.hidden}, dtype)),
    };
    block->qkv.outs = {width, kv_width, kv_width};
    block->o = dense_linear(
        upload(pattern(static_cast<std::size_t>(geometry.hidden * width), 0.067f, 4.0f),
               {geometry.hidden, width}, dtype));
    auto norm_values = pattern(static_cast<std::size_t>(geometry.hidden), 0.1f, 6.0f);
    for (auto& value : norm_values) value += 1.0f;
    block->attn_norm = upload(norm_values, {geometry.hidden}, tb::kFloat32);
    block->ffn_norm = block->attn_norm;
    block->q_norm = upload(pattern(static_cast<std::size_t>(geometry.head_dim), 0.2f, 7.0f),
                           {geometry.head_dim}, tb::kFloat32);
    block->k_norm = upload(pattern(static_cast<std::size_t>(geometry.head_dim), 0.2f, 8.0f),
                           {geometry.head_dim}, tb::kFloat32);
    block->layer = 0;
    block->attention_heads = geometry.heads;
    block->kv_heads = geometry.kv_heads;
    block->attention_head_dim = geometry.head_dim;
    block->max_position_embeddings = geometry.max_positions;
    block->rms_norm_eps = geometry.eps;
    block->norm_weight_offset = geometry.weight_offset;
    block->attention_scale = 0.0;
    block->official_bf16 = dtype == tb::kBFloat16;
    block->attention_rope = RopeCache(geometry.max_positions, geometry.rotary_dim, 10000.0);
    // The FFN is intentionally inert: this test isolates the attention span.
    const auto intermediate = geometry.hidden;
    block->ffn.gate_up.layers = {
        dense_linear(upload(std::vector<float>(std::size_t(geometry.hidden) * intermediate, 0.0f),
                            {intermediate, geometry.hidden}, dtype)),
        dense_linear(upload(std::vector<float>(std::size_t(geometry.hidden) * intermediate, 0.0f),
                            {intermediate, geometry.hidden}, dtype)),
    };
    block->ffn.gate_up.outs = {intermediate, intermediate};
    block->ffn.down = dense_linear(
        upload(std::vector<float>(std::size_t(geometry.hidden) * intermediate, 0.0f),
               {geometry.hidden, intermediate}, dtype));
    return block;
}

struct BlockRun {
    std::vector<float> output;
    std::vector<float> cache_k, cache_v;
    bool fused_span_used = false;
    bool fused_rotary_span_used = false;
    bool separate_span_used = false;
};

BlockRun run_block(const BlockGeometry& geometry, const std::vector<float>& input_values,
                   std::int64_t tokens, std::int64_t cache_position, bool fused_norm,
                   bool fused_rotary) {
    set_environment("MFQ_MINICPM_FUSED_QK_NORM_ROPE_KV", fused_norm);
    set_environment("MFQ_MINICPM_FUSED_ROPE_KV", fused_rotary);
    CudaExecutionContext execution;
    execution.profiler.enabled = true;
    execution.profiler.filter =
        "full.q_rope,full.k_rope,full.kv_write,full.qk_norm_rope_kv_write,full.rope_kv_write,"
        "full.q_view,full.k_view,full.v_view,full.qk_norm";
    auto block = make_block(geometry);
    const auto input = upload(input_values, {1, tokens, geometry.hidden}, geometry.dtype);
    auto positions = tb::arange(tokens, tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64));
    positions = (positions + cache_position).to(tb::kInt64).contiguous();
    BlockRun run;
    run.output = host_values(
        block->forward(execution, input, positions, cache_position, mfq_nullopt,
                       block->attention_rope));
    run.cache_k = host_values(block->cache.k);
    run.cache_v = host_values(block->cache.v);
    const auto& stats = execution.profiler.stats;
    run.fused_span_used = stats.contains("full.qk_norm_rope_kv_write");
    run.fused_rotary_span_used = stats.contains("full.rope_kv_write");
    run.separate_span_used = stats.contains("full.q_rope") || stats.contains("full.k_rope") ||
                             stats.contains("full.kv_write");
    execution.profiler.reset();
    return run;
}

void check_block_integration(const BlockGeometry& geometry, std::int64_t tokens,
                             std::int64_t cache_position) {
    const auto input_values =
        pattern(static_cast<std::size_t>(tokens * geometry.hidden), 0.4f, 11.0f);
    const auto separate = run_block(geometry, input_values, tokens, cache_position, false, false);
    const auto fused = run_block(geometry, input_values, tokens, cache_position, true, true);
    const auto rotary_only = run_block(geometry, input_values, tokens, cache_position, false, true);
    const auto label = "block tokens=" + std::to_string(tokens) +
                       " cache_pos=" + std::to_string(cache_position) +
                       (geometry.dtype == tb::kBFloat16 ? " bf16" : " f16");
    require(fused.fused_span_used, label + " did not take the fused span");
    require(separate.separate_span_used, label + " did not take the separate span");
    require(!fused.separate_span_used, label + " ran the separate span as well");
    // The separate norm stage with a fused rotary/cache span is the other
    // supported combination and must agree with the fully separate path. FP16
    // norm output stays FP32, so that span keeps the separate rotary for FP16.
    require(!rotary_only.fused_span_used, label + " fused the norm span as well");
    require(rotary_only.fused_rotary_span_used == (geometry.dtype == tb::kBFloat16),
            label + " rotary-only fused span selection");
    const auto tolerance = integration_tolerance(geometry.dtype);
    std::vector<double> expected(separate.output.begin(), separate.output.end());
    require_close(fused.output, expected, tolerance, label + " output");
    std::vector<double> expected_k(separate.cache_k.begin(), separate.cache_k.end());
    std::vector<double> expected_v(separate.cache_v.begin(), separate.cache_v.end());
    require_close(fused.cache_k, expected_k, tolerance, label + " key cache");
    require_close(fused.cache_v, expected_v, tolerance, label + " value cache");
    std::vector<double> rotary_output(separate.output.begin(), separate.output.end());
    require_close(rotary_only.output, rotary_output, tolerance, label + " rotary-only output");
    std::vector<double> rotary_k(separate.cache_k.begin(), separate.cache_k.end());
    require_close(rotary_only.cache_k, rotary_k, tolerance, label + " rotary-only key cache");
}

// Device time and launch count of one profiler section.
struct SectionSample {
    double milliseconds = 0.0;
    std::int64_t calls = 0;
};

// Resolve the profiler's pending event pairs into per-section totals without
// printing a report, so repeated samples can be reduced to their minimum.
std::map<std::string, SectionSample> collect_sections(CudaProfiler& profiler) {
    std::map<std::string, SectionSample> totals;
    for (auto& event : profiler.pending) {
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, event.start, event.stop);
        cudaEventDestroy(event.start);
        cudaEventDestroy(event.stop);
        auto& sample = totals[event.name];
        sample.milliseconds += double(milliseconds);
        sample.calls += 1;
    }
    profiler.pending.clear();
    return totals;
}

// Per-section device time and launch counts come from the profiler; the whole
// block is timed separately so launch latency is included once. Both take the
// minimum of repeated samples because the host is shared while measuring.
void measure_span(const BlockGeometry& geometry, std::int64_t tokens, bool fused,
                  const std::string& phase) {
    const auto span = fused ? "fused" : "separate";
    set_environment("MFQ_MINICPM_FUSED_QK_NORM_ROPE_KV", fused);
    set_environment("MFQ_MINICPM_FUSED_ROPE_KV", fused);
    CudaExecutionContext execution;
    auto block = make_block(geometry);
    const auto input_values =
        pattern(static_cast<std::size_t>(tokens * geometry.hidden), 0.4f, 13.0f);
    const auto input = upload(input_values, {1, tokens, geometry.hidden}, geometry.dtype);
    auto positions = tb::arange(tokens, tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64));
    const auto forward = [&]() {
        return block->forward(execution, input, positions, 0, mfq_nullopt, block->attention_rope);
    };
    for (int iteration = 0; iteration < 8; ++iteration) forward();

    execution.profiler.enabled = true;
    execution.profiler.filter =
        "full.q_view,full.k_view,full.v_view,full.qk_norm,full.q_rope,full.k_rope,full.kv_write,"
        "full.rope_kv_write,full.qk_norm_rope_kv_write";
    constexpr int profiled = 50;
    std::map<std::string, SectionSample> totals;
    for (int round = 0; round < profiled; ++round) forward();
    auto sample = collect_sections(execution.profiler);
    execution.profiler.enabled = false;
    for (const auto& [name, value] : sample)
        std::cout << "bench span=" << span << " phase=" << phase << " item=" << name
                  << " calls=" << value.calls << " us_per_call="
                  << value.milliseconds * 1000.0 / double(value.calls) << '\n';

    constexpr int iterations = 100;
    double best_per_call = 0.0;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int round = 0; round < 5; ++round) {
        cudaEventRecord(begin);
        for (int iteration = 0; iteration < iterations; ++iteration) forward();
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, begin, end);
        const auto per_call = double(elapsed) * 1000.0 / double(iterations);
        if (best_per_call == 0.0 || per_call < best_per_call) best_per_call = per_call;
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    std::cout << "bench span=" << span << " phase=" << phase << " item=full.forward us_per_call="
              << best_per_call << '\n';
}

void run_benchmark() {
    BlockGeometry decode;
    decode.heads = 32;
    decode.kv_heads = 8;
    decode.head_dim = 128;
    decode.rotary_dim = 128;
    decode.dtype = tb::kBFloat16;
    BlockGeometry prefill = decode;
    prefill.max_positions = 512;
    std::cout << "bench geometry heads=32 kv_heads=8 head_dim=128 bf16\n";
    measure_span(decode, 1, false, "decode");
    measure_span(decode, 1, true, "decode");
    measure_span(prefill, 128, false, "prefill");
    measure_span(prefill, 128, true, "prefill");
}

void check_operator() {
    for (const auto& spec : operator_cases()) {
        check_operator_case(spec);
        std::cout << "qk_rope_kv_test PASS case=" << spec.name << '\n';
    }
}

void check_full_block() {
    for (auto dtype : {tb::kFloat16, tb::kBFloat16}) {
        BlockGeometry geometry;
        geometry.dtype = dtype;
        check_block_integration(geometry, 1, 0);
        check_block_integration(geometry, 4, 0);
        check_block_integration(geometry, 1, 4);
        BlockGeometry partial = geometry;
        partial.head_dim = 64;
        partial.rotary_dim = 32;
        partial.weight_offset = 0.0;
        partial.heads = 8;
        partial.kv_heads = 2;
        check_block_integration(partial, 3, 0);
        std::cout << "qk_rope_kv_test PASS block dtype="
                  << (dtype == tb::kBFloat16 ? "bf16" : "f16") << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--bench") {
            run_benchmark();
            return 0;
        }
        check_operator();
        check_full_block();
        std::cout << "qk_rope_kv_test PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qk_rope_kv_test FAIL: " << error.what() << '\n';
        return 1;
    }
}
