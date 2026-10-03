#include "mlx_sparse_attention.h"

#include "mfq_mfe_prefill_embedded.h"
#include "mlx_platform.h"

#include <mlx/backend/metal/device.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;
using Kernel = mlx::core::fast::CustomKernelFunction;
using TemplateArgs = std::vector<
    std::pair<std::string, mlx::core::fast::TemplateArg>>;

#include "../kernels/mfq_sparse_attention_kernels.inc"

Kernel make_sparse_kernel(
    const char* name,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    const char* source,
    const char* header = "") {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        name,
        std::move(inputs),
        std::move(outputs),
        source,
        header,
        true,
        false,
        options);
}

const Kernel& deepselect_topk512_kernel() {
    static const auto kernel = make_sparse_kernel(
        "mfq_cpp_deepselect_topk512",
        {"x", "valid_keys", "topk_params"},
        {"out"},
        kDeepSelectTopkSource,
        kDeepSelectTopkHeader);
    return kernel;
}

const Kernel& sparse_selected_mla_short_kernel() {
    static const auto kernel = make_sparse_kernel(
        "mfq_cpp_sparse_selected_mla_short",
        {"q", "kv", "indices", "mask", "sinks", "params"},
        {"out"},
        kSparseAttentionSource);
    return kernel;
}

const Kernel& sparse_selected_mla_decode_kernel() {
    static const auto kernel = make_sparse_kernel(
        "mfq_cpp_sparse_selected_mla_decode",
        {"q", "kv", "indices", "mask", "sinks", "params"},
        {"out"},
        kSparseAttentionDecodeSource);
    return kernel;
}

const Kernel& sparse_block_gqa_gather_kernel() {
    // Singleton decode benefits from MLX's split-K SDPA rather than the
    // prefill kernel's one threadgroup per KV head. Gather chronological
    // selected blocks and the incomplete causal tail first, as in oMLX's
    // contiguous_causal_gathered_qsa_decode (see NOTICE).
    static const auto kernel = make_sparse_kernel(
        "mfq_cpp_sparse_block_gqa_gather",
        {"keys", "values", "blocks", "params"},
        {"selected_keys", "selected_values", "valid"},
        R"METAL(
            uint index = thread_position_in_grid.x;
            uint key_count = uint(params[0]);
            uint selected = uint(params[1]);
            uint valid_blocks = uint(params[2]);
            uint complete = key_count / uint(BLOCK_SIZE);
            if (index >= uint(KV_HEADS) * selected * uint(DIM)) return;
            uint dim = index % uint(DIM);
            uint token = (index / uint(DIM)) % selected;
            uint head = index / (selected * uint(DIM));
            long source = token < valid_blocks * uint(BLOCK_SIZE)
                ? long(blocks[token / uint(BLOCK_SIZE)]) * long(BLOCK_SIZE) +
                    long(token % uint(BLOCK_SIZE))
                : long(complete * uint(BLOCK_SIZE)) +
                    long(token - valid_blocks * uint(BLOCK_SIZE));
            bool present = source >= 0 && source < long(key_count);
            uint offset = uint(head * key_count * uint(DIM) +
                uint(present ? source : 0) * uint(DIM) + dim);
            selected_keys[index] = present ? keys[offset] : 0;
            selected_values[index] = present ? values[offset] : 0;
            if (head == 0 && dim == 0) valid[token] = present;
        )METAL");
    return kernel;
}

const Kernel& sparse_circular_mla_decode_kernel() {
    static const auto kernel = make_sparse_kernel(
        "mfq_cpp_sparse_circular_mla_decode",
        {
            "q",
            "local_kv",
            "pooled_kv",
            "topk",
            "sinks",
            "params",
            "decode_params",
        },
        {"out"},
        kSparseAttentionDirectDecodeSource);
    return kernel;
}

array typed_contiguous(const array& input, Dtype dtype) {
    auto result = input;
    if (result.dtype() != dtype) {
        result = mlx::core::astype(result, dtype);
    }
    return mlx::core::contiguous(result);
}

array generic_selected_mla_attention(
    const array& query,
    const array& cache,
    const array& indices,
    const array& mask,
    const array& sinks,
    float scale) {
    const int batch = query.shape(0);
    const int heads = query.shape(1);
    const int tokens = query.shape(2);
    const int selected = indices.shape(2);
    const int dimension = query.shape(3);
    auto safe_indices = mlx::core::maximum(
        indices, array(0, mlx::core::int32));
    auto expanded_cache = mlx::core::broadcast_to(
        mlx::core::expand_dims(cache, 1),
        Shape{batch, tokens, cache.shape(1), dimension});
    auto expanded_indices = mlx::core::broadcast_to(
        mlx::core::expand_dims(safe_indices, -1),
        Shape{batch, tokens, selected, dimension});
    auto gathered = mlx::core::astype(
        mlx::core::take_along_axis(
            expanded_cache, expanded_indices, 2),
        mlx::core::float32);
    auto query_values = mlx::core::astype(
        mlx::core::transpose(query, {0, 2, 1, 3}),
        mlx::core::float32);
    auto scores = mlx::core::sum(
        mlx::core::expand_dims(query_values, 3) *
            mlx::core::expand_dims(gathered, 2),
        -1) * scale;
    scores = scores + mlx::core::expand_dims(
        mlx::core::astype(mask, mlx::core::float32), 2);
    auto sink_values = mlx::core::reshape(
        mlx::core::astype(sinks, mlx::core::float32),
        Shape{1, 1, heads});
    auto maximum = mlx::core::maximum(
        mlx::core::max(scores, -1), sink_values);
    auto exponentials = mlx::core::exp(
        scores - mlx::core::expand_dims(maximum, -1));
    auto denominator = mlx::core::sum(exponentials, -1) +
        mlx::core::exp(sink_values - maximum);
    auto probabilities = exponentials /
        mlx::core::expand_dims(denominator, -1);
    return mlx::core::sum(
        mlx::core::expand_dims(probabilities, -1) *
            mlx::core::expand_dims(gathered, 2),
        3);
}

int checked_grid_product(
    std::initializer_list<int> factors,
    const char* label) {
    std::int64_t product = 1;
    for (const int factor : factors) {
        if (factor < 0 ||
            (factor != 0 &&
             product > std::numeric_limits<int>::max() / factor)) {
            throw std::invalid_argument(
                std::string(label) + " exceeds MLX grid limits");
        }
        product *= factor;
    }
    return static_cast<int>(product);
}

struct SparseBlockGqaParams {
    std::int32_t batch = 0;
    std::int32_t query_heads = 0;
    std::int32_t kv_heads = 0;
    std::int32_t queries = 0;
    std::int32_t keys = 0;
    std::int32_t selected_blocks = 0;
    std::int32_t gqa_factor = 0;
    std::int32_t query_offset = 0;
    std::int32_t block_size = 0;
    float scale = 0.0f;
    std::int64_t query_strides[3]{};
    std::int64_t key_strides[3]{};
    std::int64_t value_strides[3]{};
    std::int64_t block_strides[3]{};
};

struct SparseSelectedMlaParams {
    std::int32_t batch = 0;
    std::int32_t queries = 0;
    std::int32_t keys = 0;
    std::int32_t selected = 0;
    float scale = 0.0f;
};

struct SparseCircularMlaParams {
    std::int32_t batch = 0;
    std::int32_t queries = 0;
    std::int32_t local_length = 0;
    std::int32_t pool_capacity = 0;
    std::int32_t pool_length = 0;
    std::int32_t topk = 0;
    std::int32_t local_window = 0;
    std::int32_t pool_ratio = 0;
    std::int32_t query_offset = 0;
    float scale = 0.0f;
};

class SparseBlockGqaPrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    SparseBlockGqaPrimitive(
        mlx::core::Stream stream,
        SparseBlockGqaParams params,
        mlx::core::Dtype dtype)
        : UnaryPrimitive(stream),
          params_(params),
          dtype_(dtype) {}

    void eval_cpu(const std::vector<array>&, array&) override {
        throw std::runtime_error(
            "selected-block sparse GQA has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 4) {
            throw std::logic_error(
                "selected-block sparse GQA input count mismatch");
        }
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(selected_stream.device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            "mfq_sparse_attention_v1",
            options,
            [] {
                std::string source;
                source.reserve(
                    sizeof(detail::kSteelAttentionSource)
                    + sizeof(detail::kDsaSparsePrefillSource)
                    + sizeof(detail::kSparseBlockGqaSource)
                    + 192);
                source += "#include <metal_stdlib>\n";
                source += "#include <metal_simdgroup>\n";
                source += "#include <metal_simdgroup_matrix>\n";
                source += "using namespace metal;\n";
                source += "using bfloat16_t = bfloat;\n";
                source += detail::kSteelAttentionSource;
                source += detail::kDsaSparsePrefillSource;
                source += detail::kSparseBlockGqaSource;
                return source;
            });
        const char* kernel_name = dtype_ == mlx::core::float16
            ? "mfq_sparse_block_gqa_f16_bk64_dc64_gqa12_d256_wm2"
            : "mfq_sparse_block_gqa_bf16_bk64_dc64_gqa12_d256_wm2";
        auto* kernel = device.get_kernel(kernel_name, library);
        auto& encoder = mlx::core::metal::get_command_encoder(selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 4; ++index) {
            encoder.set_input_array(inputs[static_cast<std::size_t>(index)], index);
        }
        encoder.set_output_array(output, 4);
        encoder.set_bytes(params_, 5);
        encoder.dispatch_threadgroups(
            MTL::Size(params_.queries, params_.kv_heads, params_.batch),
            MTL::Size(32, 2, 1));
    }

    const char* name() const override {
        return "SparseBlockGqaPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const SparseBlockGqaPrimitive*>(&other);
        return primitive != nullptr
            && primitive->dtype_ == dtype_
            && primitive->params_.batch == params_.batch
            && primitive->params_.query_heads == params_.query_heads
            && primitive->params_.kv_heads == params_.kv_heads
            && primitive->params_.queries == params_.queries
            && primitive->params_.keys == params_.keys
            && primitive->params_.selected_blocks == params_.selected_blocks
            && primitive->params_.query_offset == params_.query_offset
            && primitive->params_.block_size == params_.block_size
            && primitive->params_.scale == params_.scale;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {Shape{
            params_.batch,
            params_.queries,
            params_.query_heads,
            256,
        }};
    }

private:
    SparseBlockGqaParams params_;
    mlx::core::Dtype dtype_;
};

class SparseSelectedMlaPrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    SparseSelectedMlaPrimitive(
        mlx::core::Stream stream,
        SparseSelectedMlaParams params,
        Dtype dtype)
        : UnaryPrimitive(stream), params_(params), dtype_(dtype) {}

    void eval_cpu(const std::vector<array>&, array&) override {
        throw std::runtime_error(
            "selected-token sparse MLA has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 5) {
            throw std::logic_error(
                "selected-token sparse MLA input count mismatch");
        }
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(selected_stream.device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            "mfq_sparse_attention_v1",
            options,
            [] {
                std::string source;
                source.reserve(
                    sizeof(detail::kSteelAttentionSource)
                    + sizeof(detail::kDsaSparsePrefillSource)
                    + sizeof(detail::kSparseBlockGqaSource)
                    + 192);
                source += "#include <metal_stdlib>\n";
                source += "#include <metal_simdgroup>\n";
                source += "#include <metal_simdgroup_matrix>\n";
                source += "using namespace metal;\n";
                source += "using bfloat16_t = bfloat;\n";
                source += detail::kSteelAttentionSource;
                source += detail::kDsaSparsePrefillSource;
                source += detail::kSparseBlockGqaSource;
                return source;
            });
        const char* kernel_name = dtype_ == mlx::core::bfloat16
            ? "mfq_dsa_sparse_prefill_bf16_bk256_dc32"
            : "mfq_dsa_sparse_prefill_f16_bk256_dc32";
        auto* kernel = device.get_kernel(kernel_name, library);
        auto& encoder =
            mlx::core::metal::get_command_encoder(selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 5; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)], index);
        }
        encoder.set_output_array(output, 5);
        encoder.set_bytes(params_, 6);
        encoder.dispatch_threadgroups(
            MTL::Size(params_.queries, params_.batch, 1),
            MTL::Size(32, 8, 1));
    }

    const char* name() const override {
        return "SparseSelectedMlaPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive = dynamic_cast<
            const SparseSelectedMlaPrimitive*>(&other);
        return primitive != nullptr
            && primitive->dtype_ == dtype_
            && primitive->params_.batch == params_.batch
            && primitive->params_.queries == params_.queries
            && primitive->params_.keys == params_.keys
            && primitive->params_.selected == params_.selected
            && primitive->params_.scale == params_.scale;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {Shape{
            params_.batch,
            params_.queries,
            64,
            512,
        }};
    }

private:
    SparseSelectedMlaParams params_;
    Dtype dtype_;
};

class SparseCircularMlaPrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    SparseCircularMlaPrimitive(
        mlx::core::Stream stream,
        SparseCircularMlaParams params,
        Dtype dtype)
        : UnaryPrimitive(stream), params_(params), dtype_(dtype) {}

    void eval_cpu(const std::vector<array>&, array&) override {
        throw std::runtime_error(
            "circular sparse MLA has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 5) {
            throw std::logic_error(
                "circular sparse MLA input count mismatch");
        }
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(selected_stream.device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            "mfq_sparse_attention_v1",
            options,
            [] {
                std::string source;
                source.reserve(
                    sizeof(detail::kSteelAttentionSource)
                    + sizeof(detail::kDsaSparsePrefillSource)
                    + sizeof(detail::kSparseBlockGqaSource)
                    + 192);
                source += "#include <metal_stdlib>\n";
                source += "#include <metal_simdgroup>\n";
                source += "#include <metal_simdgroup_matrix>\n";
                source += "using namespace metal;\n";
                source += "using bfloat16_t = bfloat;\n";
                source += detail::kSteelAttentionSource;
                source += detail::kDsaSparsePrefillSource;
                source += detail::kSparseBlockGqaSource;
                return source;
            });
        const char* kernel_name = dtype_ == mlx::core::bfloat16
            ? "mfq_dsa_sparse_circular_bf16_bk256_dc32"
            : "mfq_dsa_sparse_circular_f16_bk256_dc32";
        auto* kernel = device.get_kernel(kernel_name, library);
        auto& encoder =
            mlx::core::metal::get_command_encoder(selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 5; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)], index);
        }
        encoder.set_output_array(output, 5);
        encoder.set_bytes(params_, 6);
        encoder.dispatch_threadgroups(
            MTL::Size(params_.queries, params_.batch, 1),
            MTL::Size(32, 8, 1));
    }

    const char* name() const override {
        return "SparseCircularMlaPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive = dynamic_cast<
            const SparseCircularMlaPrimitive*>(&other);
        return primitive != nullptr
            && primitive->dtype_ == dtype_
            && primitive->params_.batch == params_.batch
            && primitive->params_.queries == params_.queries
            && primitive->params_.local_length == params_.local_length
            && primitive->params_.pool_capacity == params_.pool_capacity
            && primitive->params_.pool_length == params_.pool_length
            && primitive->params_.topk == params_.topk
            && primitive->params_.local_window == params_.local_window
            && primitive->params_.pool_ratio == params_.pool_ratio
            && primitive->params_.query_offset == params_.query_offset
            && primitive->params_.scale == params_.scale;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {Shape{
            params_.batch,
            params_.queries,
            64,
            512,
        }};
    }

private:
    SparseCircularMlaParams params_;
    Dtype dtype_;
};

} // namespace

bool mlx_deepselect_topk512_preferred(int width, int rows) noexcept {
    const bool favorable_shape =
        (width >= 16384 && rows >= 64) ||
        (width >= 8192 && rows >= 128);
    if (!favorable_shape) {
        return false;
    }
    if (const auto* requested = std::getenv(
            "MFQ_METAL_DEEPSELECT")) {
        return std::strcmp(requested, "0") != 0 &&
            std::strcmp(requested, "false") != 0 &&
            std::strcmp(requested, "off") != 0;
    }
    return mlx_apple_chip_is("Apple M3 Ultra");
}

array mlx_deepselect_topk512(
    const array& scores,
    const std::optional<array>& valid_keys) {
    auto source = typed_contiguous(scores, mlx::core::float32);
    if (source.ndim() != 3 || source.shape(0) <= 0 ||
        source.shape(1) <= 0 || source.shape(2) < 512) {
        throw std::invalid_argument(
            "DeepSelect top-k expects f32 [B,M,K>=512]");
    }
    const int batch = source.shape(0);
    const int queries = source.shape(1);
    const int keys = source.shape(2);
    array counts = valid_keys.has_value()
        ? typed_contiguous(*valid_keys, mlx::core::int32)
        : mlx::core::full(
              Shape{batch, queries}, keys, mlx::core::int32);
    if (counts.shape() != Shape{batch, queries}) {
        throw std::invalid_argument(
            "DeepSelect valid-key counts must be [B,M]");
    }
    const int rows = checked_grid_product(
        {batch, queries}, "DeepSelect top-k row count");
    const int grid = checked_grid_product(
        {rows, 1024}, "DeepSelect top-k grid");
    const array topk_params({keys}, mlx::core::int32);
    auto outputs = deepselect_topk512_kernel()(
        {source, counts, topk_params},
        {Shape{batch, queries, 512}},
        {mlx::core::int32},
        {grid, 1, 1},
        {1024, 1, 1},
        {},
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array mlx_sparse_block_gqa_attention(
    const array& query,
    const array& key,
    const array& value,
    const array& selected_blocks,
    int query_offset,
    int block_size,
    std::optional<float> scale) {
    const Dtype attention_dtype =
        query.dtype() == mlx::core::bfloat16 &&
        key.dtype() == mlx::core::bfloat16 &&
        value.dtype() == mlx::core::bfloat16
            ? mlx::core::bfloat16
            : mlx::core::float16;
    auto selected_query = typed_contiguous(query, attention_dtype);
    auto selected_key = typed_contiguous(key, attention_dtype);
    auto selected_value = typed_contiguous(value, attention_dtype);
    auto blocks = typed_contiguous(selected_blocks, mlx::core::int32);
    if (selected_query.ndim() != 4 || selected_key.ndim() != 4 ||
        selected_value.shape() != selected_key.shape() || blocks.ndim() != 3 ||
        selected_query.shape(0) != selected_key.shape(0) ||
        selected_query.shape(0) != blocks.shape(0) ||
        selected_query.shape(2) != blocks.shape(1) ||
        selected_query.shape(1) != 24 || selected_key.shape(1) != 2 ||
        selected_query.shape(3) != 256 || selected_key.shape(3) != 256 ||
        blocks.shape(2) <= 0 || query_offset < 0 ||
        query_offset + selected_query.shape(2) > selected_key.shape(2) ||
        block_size <= 0) {
        throw std::invalid_argument(
            "unsupported selected-block sparse GQA geometry");
    }
    const float selected_scale = scale.value_or(1.0f / std::sqrt(256.0f));
    if (!std::isfinite(selected_scale)) {
        throw std::invalid_argument(
            "selected-block sparse GQA scale must be finite");
    }

    const char* gather_setting = std::getenv("MFQ_METAL_SPARSE_GQA_DECODE_GATHER");
    if (selected_query.shape(0) == 1 && selected_query.shape(2) == 1 &&
        query_offset + 1 == selected_key.shape(2) &&
        (gather_setting == nullptr || std::string(gather_setting) != "0")) {
        const int complete = selected_key.shape(2) / block_size;
        const int valid_blocks = std::min(blocks.shape(2), complete);
        const int selected = valid_blocks * block_size +
            selected_key.shape(2) % block_size;
        if (selected > 0) {
            const Shape gathered_shape{1, selected_key.shape(1), selected, 256};
            // Context length changes every token; keep it in runtime data,
            // never in the kernel template/cache key.
            const array gather_params(
                {selected_key.shape(2), selected, valid_blocks}, mlx::core::int32);
            auto gathered = sparse_block_gqa_gather_kernel()(
                {selected_key, selected_value, blocks, gather_params},
                {gathered_shape, gathered_shape, Shape{1, 1, 1, selected}},
                {attention_dtype, attention_dtype, mlx::core::bool_},
                {selected_key.shape(1) * selected * 256, 1, 1},
                {256, 1, 1},
                {
                    {"KV_HEADS", selected_key.shape(1)},
                    {"DIM", 256},
                    {"BLOCK_SIZE", block_size},
                },
                std::nullopt,
                false,
                {});
            auto output = mlx::core::fast::scaled_dot_product_attention(
                selected_query, gathered[0], gathered[1], selected_scale,
                "", gathered[2]);
            return mlx::core::transpose(output, {0, 2, 1, 3});
        }
    }

    SparseBlockGqaParams params{
        .batch = selected_query.shape(0),
        .query_heads = selected_query.shape(1),
        .kv_heads = selected_key.shape(1),
        .queries = selected_query.shape(2),
        .keys = selected_key.shape(2),
        .selected_blocks = blocks.shape(2),
        .gqa_factor = selected_query.shape(1) / selected_key.shape(1),
        .query_offset = query_offset,
        .block_size = block_size,
        .scale = selected_scale,
        .query_strides = {
            selected_query.strides(0),
            selected_query.strides(1),
            selected_query.strides(2)},
        .key_strides = {
            selected_key.strides(0),
            selected_key.strides(1),
            selected_key.strides(2)},
        .value_strides = {
            selected_value.strides(0),
            selected_value.strides(1),
            selected_value.strides(2)},
        .block_strides = {
            blocks.strides(0),
            blocks.strides(1),
            blocks.strides(2)},
    };
    auto stream = mlx::core::default_stream(mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "selected-block sparse GQA requires Metal");
    }
    return array(
        Shape{
            params.batch,
            params.queries,
            params.query_heads,
            256,
        },
        attention_dtype,
        std::make_shared<SparseBlockGqaPrimitive>(
            stream,
            params,
            attention_dtype),
        std::vector<array>{
            std::move(selected_query),
            std::move(selected_key),
            std::move(selected_value),
            std::move(blocks),
        });
}

array mlx_sparse_selected_mla_attention(
    const array& query,
    const array& kv_cache,
    const array& selected_indices,
    const array& selected_mask,
    const array& sinks,
    std::optional<float> scale) {
    const Dtype cache_dtype = kv_cache.dtype() == mlx::core::bfloat16
        ? mlx::core::bfloat16
        : mlx::core::float16;
    auto selected_query = typed_contiguous(query, mlx::core::float32);
    auto selected_cache = typed_contiguous(kv_cache, cache_dtype);
    auto indices = typed_contiguous(selected_indices, mlx::core::int32);
    auto mask = typed_contiguous(selected_mask, mlx::core::float16);
    auto sink_logits = typed_contiguous(sinks, mlx::core::float32);
    if (selected_query.ndim() != 4 || selected_query.shape(0) <= 0 ||
        selected_query.shape(1) <= 0 || selected_query.shape(2) <= 0 ||
        selected_query.shape(3) <= 0 || selected_cache.ndim() != 3 ||
        selected_cache.shape(0) != selected_query.shape(0) ||
        selected_cache.shape(1) <= 0 ||
        selected_cache.shape(2) != selected_query.shape(3) || indices.ndim() != 3 ||
        indices.shape(0) != selected_query.shape(0) ||
        indices.shape(1) != selected_query.shape(2) ||
        indices.shape(2) <= 0 || indices.shape(2) % 32 != 0 ||
        mask.shape() != indices.shape() ||
        sink_logits.size() != static_cast<std::size_t>(selected_query.shape(1))) {
        throw std::invalid_argument(
            "unsupported selected-token sparse MLA geometry");
    }
    const int heads = selected_query.shape(1);
    const int dimension = selected_query.shape(3);
    const float selected_scale = scale.value_or(
        1.0f / std::sqrt(static_cast<float>(dimension)));
    if (!std::isfinite(selected_scale) || selected_scale <= 0.0f) {
        throw std::invalid_argument(
            "selected-token sparse MLA scale must be finite and positive");
    }
    if (heads != 64 || dimension != 512) {
        return generic_selected_mla_attention(
            selected_query,
            selected_cache,
            indices,
            mask,
            sink_logits,
            selected_scale);
    }
    constexpr int kHeads = 64;
    constexpr int kDimension = 512;
    SparseSelectedMlaParams params{
        .batch = selected_query.shape(0),
        .queries = selected_query.shape(2),
        .keys = selected_cache.shape(1),
        .selected = indices.shape(2),
        .scale = selected_scale,
    };
    if (params.queries < 32) {
        const array scale_parameter({selected_scale}, mlx::core::float32);
        const Shape output_shape{
            params.batch,
            params.queries,
            kHeads,
            kDimension,
        };
        const TemplateArgs templates{
            {"B", params.batch},
            {"M", params.queries},
            {"MAX_SEQ", params.keys},
            {"SELECTED", params.selected},
        };
        // The decode kernel already maps query rows independently and keeps
        // the same 32-lane dot/softmax/value reduction as M=1. Use it for
        // DSpark's M=2..6 verifier blocks as well; the former short-prefill
        // kernel used eight times as many threads and changed reduction order.
        const bool decode_consistent = params.queries <= 6;
        const int grid = decode_consistent
            ? checked_grid_product(
                {params.batch, params.queries, 16, 128},
                "selected-token sparse MLA decode grid")
            : checked_grid_product(
                {params.batch, params.queries, kHeads, 256},
                "selected-token sparse MLA short-query grid");
        auto outputs = (decode_consistent
            ? sparse_selected_mla_decode_kernel()
            : sparse_selected_mla_short_kernel())(
                {
                    selected_query,
                    selected_cache,
                    indices,
                    mask,
                    sink_logits,
                    scale_parameter,
                },
                {output_shape},
                {mlx::core::float32},
                {grid, 1, 1},
                {decode_consistent ? 128 : 256, 1, 1},
                templates,
                std::nullopt,
                false,
                {});
        return std::move(outputs.front());
    }
    auto attention_query = typed_contiguous(selected_query, cache_dtype);
    auto attention_sinks = typed_contiguous(sink_logits, cache_dtype);
    auto stream = mlx::core::default_stream(mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "selected-token sparse MLA requires Metal");
    }
    return array(
        Shape{params.batch, params.queries, kHeads, kDimension},
        cache_dtype,
        std::make_shared<SparseSelectedMlaPrimitive>(
            stream,
            params,
            cache_dtype),
        std::vector<array>{
            std::move(attention_query),
            std::move(selected_cache),
            std::move(indices),
            std::move(mask),
            std::move(attention_sinks),
        });
}

array mlx_sparse_circular_mla_attention(
    const array& query,
    const array& local_kv,
    const std::optional<array>& pooled_kv,
    int pool_len,
    const array& topk,
    const array& sinks,
    int query_offset,
    int pool_ratio,
    int local_window,
    std::optional<float> scale) {
    constexpr int kHeads = 64;
    constexpr int kDimension = 512;
    const Dtype cache_dtype =
        local_kv.dtype() == mlx::core::bfloat16 &&
        (!pooled_kv || pooled_kv->dtype() == mlx::core::bfloat16)
            ? mlx::core::bfloat16
            : mlx::core::float16;
    auto selected_query = typed_contiguous(query, cache_dtype);
    auto local = typed_contiguous(local_kv, cache_dtype);
    auto pool = pooled_kv
        ? typed_contiguous(*pooled_kv, cache_dtype)
        : local;
    auto selected_topk = typed_contiguous(topk, mlx::core::int32);
    auto sink_logits = typed_contiguous(sinks, cache_dtype);
    if (selected_query.ndim() != 4 || selected_query.shape(0) <= 0 ||
        selected_query.shape(1) != kHeads ||
        selected_query.shape(2) < 2 ||
        selected_query.shape(3) != kDimension || local.ndim() != 3 ||
        local.shape(0) != selected_query.shape(0) ||
        local.shape(1) < selected_query.shape(2) ||
        local.shape(2) != kDimension || pool.ndim() != 3 ||
        pool.shape(0) != selected_query.shape(0) ||
        pool.shape(2) != kDimension || pool_len < 0 ||
        pool_len > pool.shape(1) ||
        selected_topk.ndim() != 3 ||
        selected_topk.shape(0) != selected_query.shape(0) ||
        selected_topk.shape(1) != selected_query.shape(2) ||
        (pool_len > 0 && selected_topk.shape(2) <= 0) ||
        sink_logits.size() != kHeads ||
        query_offset < 0 || pool_ratio <= 0 || local_window <= 0) {
        throw std::invalid_argument(
            "unsupported circular multi-query sparse MLA geometry");
    }
    const float selected_scale = scale.value_or(
        1.0f / std::sqrt(static_cast<float>(kDimension)));
    if (!std::isfinite(selected_scale) || selected_scale <= 0.0f) {
        throw std::invalid_argument(
            "circular multi-query sparse MLA scale must be finite and positive");
    }
    SparseCircularMlaParams params{
        .batch = selected_query.shape(0),
        .queries = selected_query.shape(2),
        .local_length = local.shape(1),
        .pool_capacity = pool.shape(1),
        .pool_length = pool_len,
        .topk = selected_topk.shape(2),
        .local_window = local_window,
        .pool_ratio = pool_ratio,
        .query_offset = query_offset,
        .scale = selected_scale,
    };
    auto stream = mlx::core::default_stream(mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "circular multi-query sparse MLA requires Metal");
    }
    return array(
        Shape{params.batch, params.queries, kHeads, kDimension},
        cache_dtype,
        std::make_shared<SparseCircularMlaPrimitive>(
            stream,
            params,
            cache_dtype),
        std::vector<array>{
            std::move(selected_query),
            std::move(local),
            std::move(pool),
            std::move(selected_topk),
            std::move(sink_logits),
        });
}

array mlx_sparse_circular_mla_decode_attention(
    const array& query,
    const array& local_kv,
    const std::optional<array>& pooled_kv,
    int pool_len,
    const array& topk,
    const array& sinks,
    int sequence_length,
    int pool_ratio,
    int local_window,
    std::optional<float> scale) {
    constexpr int kHeads = 64;
    constexpr int kDimension = 512;
    const Dtype cache_dtype =
        local_kv.dtype() == mlx::core::bfloat16 &&
        (!pooled_kv || pooled_kv->dtype() == mlx::core::bfloat16)
            ? mlx::core::bfloat16
            : mlx::core::float16;
    auto selected_query = typed_contiguous(query, mlx::core::float32);
    auto local = typed_contiguous(local_kv, cache_dtype);
    auto selected_topk = typed_contiguous(topk, mlx::core::int32);
    auto sink_logits = typed_contiguous(sinks, mlx::core::float32);
    auto pool = pooled_kv
        ? typed_contiguous(*pooled_kv, cache_dtype)
        : local;
    if (selected_query.ndim() != 4 || selected_query.shape(0) <= 0 ||
        selected_query.shape(1) != kHeads || selected_query.shape(2) != 1 ||
        selected_query.shape(3) != kDimension ||
        local.shape() != Shape{
            selected_query.shape(0), local_window, kDimension} ||
        pool.ndim() != 3 || pool.shape(0) != selected_query.shape(0) ||
        pool.shape(2) != kDimension || pool_len < 0 ||
        pool_len > pool.shape(1) || selected_topk.ndim() != 3 ||
        selected_topk.shape(0) != selected_query.shape(0) ||
        selected_topk.shape(1) != 1 || sink_logits.size() != kHeads ||
        sequence_length <= 0 || pool_ratio <= 0 || local_window <= 0) {
        throw std::invalid_argument(
            "unsupported circular sparse MLA decode geometry");
    }
    const float selected_scale = scale.value_or(
        1.0f / std::sqrt(static_cast<float>(kDimension)));
    if (!std::isfinite(selected_scale) || selected_scale <= 0.0f) {
        throw std::invalid_argument(
            "circular sparse MLA decode scale must be finite and positive");
    }
    const int batch = selected_query.shape(0);
    const int topk_count = selected_topk.shape(2);
    const int grid = checked_grid_product(
        {batch, 16, 128},
        "circular sparse MLA decode grid");
    const array scale_parameter({selected_scale}, mlx::core::float32);
    const array decode_parameters(
        {sequence_length, pool_len, topk_count},
        mlx::core::int32);
    auto outputs = sparse_circular_mla_decode_kernel()(
        {
            selected_query,
            local,
            pool,
            selected_topk,
            sink_logits,
            scale_parameter,
            decode_parameters,
        },
        {Shape{batch, 1, kHeads, kDimension}},
        {mlx::core::float32},
        {grid, 1, 1},
        {128, 1, 1},
        {
            {"B", batch},
            {"POOL_CAPACITY", pool.shape(1)},
            {"RATIO", pool_ratio},
            {"WINDOW", local_window},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

} // namespace mfq::metal
