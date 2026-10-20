#include "mlx_paged_session_codec.h"
#include "mlx_resident_budget.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <mlx/mlx.h>

namespace mfq::metal {
namespace {

using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::array;

constexpr std::array<std::uint8_t, 8> kMagic{
    'M', 'F', 'Q', 'M', 'L', 'X', '1', 0};
constexpr std::uint32_t kVersion = 4;
constexpr std::uint8_t kKvLayer = 1;
constexpr std::uint8_t kRecurrentLayer = 2;
constexpr std::uint8_t kRecurrentUnavailableLayer = 3;
constexpr std::uint32_t kMiniRuntime = 1;
constexpr std::uint32_t kQwen35Runtime = 2;
constexpr std::uint32_t kQwen4Runtime = 3;
constexpr std::uint8_t kQsaLayer = 4;
constexpr std::uint8_t kGdnLayer = 5;
constexpr std::uint8_t kGdnUnavailableLayer = 6;

class Writer {
public:
    template <typename T>
    void scalar(T value) {
        ensure_capacity(bytes_.size() + sizeof(T));
        using Unsigned = std::make_unsigned_t<T>;
        const auto converted = static_cast<Unsigned>(value);
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            bytes_.push_back(static_cast<std::uint8_t>(
                converted >> (index * 8)));
        }
    }

    void raw(const void* data, std::size_t size) {
        ensure_capacity(bytes_.size() + size);
        const auto* begin = static_cast<const std::uint8_t*>(data);
        bytes_.insert(bytes_.end(), begin, begin + size);
    }

    void tensor(const array& source) {
        auto value = mlx::core::contiguous(source);
        value.eval();
        scalar<std::uint8_t>(
            static_cast<std::uint8_t>(value.dtype().val()));
        if (value.ndim() == 0 || value.ndim() > 8) {
            throw std::runtime_error("invalid MLX cache tensor rank");
        }
        scalar<std::uint8_t>(static_cast<std::uint8_t>(value.ndim()));
        scalar<std::uint16_t>(0);
        for (const auto dimension : value.shape()) {
            if (dimension <= 0) {
                throw std::runtime_error("invalid MLX cache tensor shape");
            }
            scalar<std::int32_t>(dimension);
        }
        scalar<std::uint64_t>(value.nbytes());
        raw(value.data<std::uint8_t>(), value.nbytes());
    }

    std::uint8_t* tensor_storage(const Shape& shape, Dtype dtype) {
        std::size_t bytes = dtype.size();
        scalar<std::uint8_t>(static_cast<std::uint8_t>(dtype.val()));
        scalar<std::uint8_t>(static_cast<std::uint8_t>(shape.size()));
        scalar<std::uint16_t>(0);
        for (const int dimension : shape) {
            if (dimension <= 0 || bytes > std::numeric_limits<std::size_t>::max() / dimension)
                throw std::runtime_error("invalid offloaded cache tensor shape");
            scalar<std::int32_t>(dimension);
            bytes *= dimension;
        }
        scalar<std::uint64_t>(bytes);
        const auto offset = bytes_.size();
        ensure_capacity(offset + bytes);
        bytes_.resize(offset + bytes);
        return bytes_.data() + offset;
    }

    void offloaded_kv_tensor(const MlxQsaKvSnapshot& state, int start, int count, bool value) {
        const int key_width = state.quantization.enabled() ? mlx_kv_packed_width(state.dimension, state.quantization.key_bits()) : state.dimension;
        const int value_width = state.quantization.enabled() ? mlx_kv_packed_width(state.dimension, state.quantization.value_bits()) : state.dimension;
        const int width = value ? value_width : key_width;
        auto* destination = tensor_storage(Shape{1, state.heads, count, width}, state.dtype);
        const auto lane = static_cast<std::size_t>(width) * state.dtype.size();
        const auto key_bytes = static_cast<std::size_t>(key_width) * state.dtype.size() * state.heads;
        std::vector<std::uint8_t> row(static_cast<std::size_t>(key_width + value_width) * state.dtype.size() * state.heads);
        for (int i = 0; i < count; ++i) {
            state.rows.read_rows(start + i, 1, row.data());
            for (int head = 0; head < state.heads; ++head)
                std::memcpy(destination + (head * static_cast<std::size_t>(count) + i) * lane,
                    row.data() + (value ? key_bytes : 0) + head * lane, lane);
        }
    }

    void offloaded_index_tensor(const MlxQsaIndexSnapshot& state, int start, int count) {
        auto* destination = tensor_storage(Shape{1, count, state.width}, state.dtype);
        state.rows.read_rows(start, count, destination);
    }

    std::shared_ptr<const std::vector<std::uint8_t>> finish() && {
        return std::make_shared<const std::vector<std::uint8_t>>(
            std::move(bytes_));
    }

private:
    void ensure_capacity(std::size_t needed) {
        if (needed <= bytes_.capacity()) return;
        const auto capacity = std::max(needed, bytes_.capacity() * 2);
        auto budget = std::make_unique<MlxResidentBudgetHold>(capacity);
        bytes_.reserve(capacity);
        budget_ = std::move(budget);
    }
    std::vector<std::uint8_t> bytes_;
    std::unique_ptr<MlxResidentBudgetHold> budget_;
};

struct SerializedTensor {
    // Non-owning view into a PagedPrefixPayload. decode() keeps every shared
    // payload alive until reconstruction completes, so block tensors can be
    // copied directly into their final full-prefix allocation instead of
    // allocating one MLX array per block and concatenating a second time.
    Dtype dtype = mlx::core::float16;
    Shape shape;
    const std::uint8_t* data = nullptr;
    std::size_t bytes = 0;
};

class Reader {
public:
    explicit Reader(const std::vector<std::uint8_t>& bytes)
        : cursor_(bytes.data()), end_(bytes.data() + bytes.size()) {}

    template <typename T>
    T scalar(const char* name) {
        if (remaining() < sizeof(T)) {
            throw std::runtime_error(
                std::string("truncated MLX cache ") + name);
        }
        using Unsigned = std::make_unsigned_t<T>;
        Unsigned converted = 0;
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            converted |= static_cast<Unsigned>(cursor_[index]) << (index * 8);
        }
        cursor_ += sizeof(T);
        return static_cast<T>(converted);
    }

    void raw(void* destination, std::size_t size, const char* name) {
        if (remaining() < size) {
            throw std::runtime_error(
                std::string("truncated MLX cache ") + name);
        }
        std::memcpy(destination, cursor_, size);
        cursor_ += size;
    }

    SerializedTensor tensor() {
        const auto dtype_value = scalar<std::uint8_t>("dtype");
        const auto rank = scalar<std::uint8_t>("rank");
        (void)scalar<std::uint16_t>("tensor flags");
        if (rank == 0 || rank > 8 ||
            dtype_value > static_cast<std::uint8_t>(Dtype::Val::complex64)) {
            throw std::runtime_error("invalid MLX cache tensor header");
        }
        Shape shape;
        shape.reserve(rank);
        std::size_t elements = 1;
        for (std::uint8_t index = 0; index < rank; ++index) {
            const auto dimension = scalar<std::int32_t>("shape");
            if (dimension <= 0 ||
                elements > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(dimension)) {
                throw std::runtime_error("invalid MLX cache tensor shape");
            }
            shape.push_back(dimension);
            elements *= static_cast<std::size_t>(dimension);
        }
        const auto dtype = dtype_from_value(
            static_cast<Dtype::Val>(dtype_value));
        const auto bytes = scalar<std::uint64_t>("tensor size");
        if (elements > std::numeric_limits<std::size_t>::max() / dtype.size() ||
            bytes != elements * dtype.size() || bytes > remaining()) {
            throw std::runtime_error("invalid MLX cache tensor size");
        }
        SerializedTensor result{
            dtype,
            std::move(shape),
            cursor_,
            static_cast<std::size_t>(bytes),
        };
        cursor_ += static_cast<std::size_t>(bytes);
        return result;
    }

    void expect_end() const {
        if (cursor_ != end_) {
            throw std::runtime_error("trailing MLX cache payload bytes");
        }
    }

    std::size_t remaining() const noexcept {
        return static_cast<std::size_t>(end_ - cursor_);
    }

private:
    static Dtype dtype_from_value(Dtype::Val value) {
        switch (value) {
            case Dtype::Val::bool_: return mlx::core::bool_;
            case Dtype::Val::uint8: return mlx::core::uint8;
            case Dtype::Val::uint16: return mlx::core::uint16;
            case Dtype::Val::uint32: return mlx::core::uint32;
            case Dtype::Val::uint64: return mlx::core::uint64;
            case Dtype::Val::int8: return mlx::core::int8;
            case Dtype::Val::int16: return mlx::core::int16;
            case Dtype::Val::int32: return mlx::core::int32;
            case Dtype::Val::int64: return mlx::core::int64;
            case Dtype::Val::float16: return mlx::core::float16;
            case Dtype::Val::float32: return mlx::core::float32;
            case Dtype::Val::float64: return mlx::core::float64;
            case Dtype::Val::bfloat16: return mlx::core::bfloat16;
            case Dtype::Val::complex64: return mlx::core::complex64;
        }
        throw std::runtime_error("unsupported MLX cache tensor dtype");
    }

    const std::uint8_t* cursor_;
    const std::uint8_t* end_;
};

struct DecodedLayer {
    std::uint8_t kind = 0;
    bool exact = false;
    int batch = 0;
    int heads = 0;
    int maximum_sequence = 0;
    int head_dimension = 0;
    int capacity = 0;
    int position = 0;
    int index_start = 0;
    int index_ratio = 0;
    Dtype dtype = mlx::core::float16;
    MlxKvQuantization quantization;
    SerializedTensor first;
    SerializedTensor second;
    std::optional<SerializedTensor> index;
    std::optional<SerializedTensor> pooled;
    std::optional<SerializedTensor> ple;
    std::vector<std::int64_t> ple_context;
};

struct DecodedBlock {
    std::uint32_t runtime = 0;
    std::uint32_t start = 0;
    std::uint32_t count = 0;
    std::vector<DecodedLayer> layers;
    std::vector<DecodedLayer> mtp_layers;
    std::optional<SerializedTensor> last_hidden;
};

void write_header(
    Writer& writer,
    std::uint32_t runtime,
    std::uint32_t start,
    std::uint32_t count,
    std::size_t layers) {
    writer.raw(kMagic.data(), kMagic.size());
    writer.scalar<std::uint32_t>(kVersion);
    writer.scalar<std::uint32_t>(runtime);
    writer.scalar<std::uint32_t>(start);
    writer.scalar<std::uint32_t>(count);
    writer.scalar<std::uint32_t>(static_cast<std::uint32_t>(layers));
}

void write_kv_layer(
    Writer& writer,
    const MlxKvCacheSnapshot& snapshot,
    int start,
    int count,
    std::uint8_t kind = kKvLayer,
    bool exact = true) {
    if (snapshot.position < start + count || snapshot.batch != 1) {
        throw std::runtime_error("invalid MLX KV cache block range");
    }
    writer.scalar<std::uint8_t>(kind);
    writer.scalar<std::uint8_t>(
        static_cast<std::uint8_t>(snapshot.dtype.val()));
    writer.scalar<std::uint16_t>((exact ? 1 : 0) | (snapshot.quantization.enabled()
        ? 2 | (static_cast<int>(snapshot.quantization.bits * 2) << 8) : 0));
    writer.scalar<std::int32_t>(snapshot.batch);
    writer.scalar<std::int32_t>(snapshot.heads);
    writer.scalar<std::int32_t>(snapshot.maximum_sequence);
    writer.scalar<std::int32_t>(snapshot.head_dimension);
    writer.scalar<std::int32_t>(snapshot.capacity);
    writer.scalar<std::int32_t>(start + count);
    const Shape begin{0, 0, start, 0};
    const Shape end{
        snapshot.batch,
        snapshot.heads,
        start + count,
        snapshot.key.shape(3),
    };
    writer.tensor(mlx::core::slice(snapshot.key, begin, end));
    writer.tensor(mlx::core::slice(snapshot.value, begin, Shape{snapshot.batch, snapshot.heads, start + count, snapshot.value.shape(3)}));
}

void write_recurrent_layer(
    Writer& writer,
    const MlxQwen35LinearAttentionCacheSnapshot& snapshot,
    int boundary,
    std::uint8_t kind = kRecurrentLayer) {
    if (snapshot.batch != 1 || snapshot.position < boundary) {
        throw std::runtime_error("invalid recurrent cache boundary");
    }
    writer.scalar<std::uint8_t>(kind);
    writer.scalar<std::uint8_t>(0);
    writer.scalar<std::uint16_t>(0);
    writer.scalar<std::int32_t>(snapshot.batch);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(boundary);
    writer.tensor(snapshot.convolution_state);
    writer.tensor(snapshot.recurrent_state);
}

void write_recurrent_unavailable_layer(
    Writer& writer,
    const MlxQwen35LinearAttentionCacheSnapshot& snapshot,
    int boundary,
    std::uint8_t kind = kRecurrentUnavailableLayer) {
    if (snapshot.batch != 1 || snapshot.position < boundary) {
        throw std::runtime_error("invalid recurrent cache boundary");
    }
    writer.scalar<std::uint8_t>(kind);
    writer.scalar<std::uint8_t>(0);
    writer.scalar<std::uint16_t>(0);
    writer.scalar<std::int32_t>(snapshot.batch);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(boundary);
}

void write_optional_tensor(Writer& writer, const std::optional<array>& value) {
    writer.scalar<std::uint8_t>(value.has_value());
    if (value) writer.tensor(*value);
}

void write_flash_layer(Writer& writer, const MlxQwen4LayerCacheSnapshot& layer,
    std::size_t start, std::size_t count, bool exact) {
    const int boundary = static_cast<int>(start + count);
    if (layer.batch != 1 || layer.position < boundary ||
        ((layer.kv || layer.offloaded_kv) ? !layer.index_keys : (!layer.convolution || !layer.recurrent)))
        throw std::runtime_error("invalid Flash-Next cache layer");
    if (layer.kv || layer.offloaded_kv) {
        if (layer.offloaded_kv) {
            const auto& kv = *layer.offloaded_kv;
            writer.scalar<std::uint8_t>(kQsaLayer);
            writer.scalar<std::uint8_t>(static_cast<std::uint8_t>(kv.dtype.val()));
            writer.scalar<std::uint16_t>((exact ? 1 : 0) | (kv.quantization.enabled()
                ? 2 | (static_cast<int>(kv.quantization.bits * 2) << 8) : 0));
            writer.scalar<std::int32_t>(1);
            writer.scalar<std::int32_t>(kv.heads);
            writer.scalar<std::int32_t>(kv.maximum);
            writer.scalar<std::int32_t>(kv.dimension);
            writer.scalar<std::int32_t>(boundary);
            writer.scalar<std::int32_t>(boundary);
            writer.offloaded_kv_tensor(kv, start, count, false);
            writer.offloaded_kv_tensor(kv, start, count, true);
        } else write_kv_layer(writer, *layer.kv, static_cast<int>(start), static_cast<int>(count), kQsaLayer, exact);
        const int ratio = layer.index_ratio;
        if (ratio <= 0 || layer.index_start < 0 || layer.index_start % ratio ||
            layer.index_keys->shape() != Shape{1, layer.position - layer.index_start, layer.index_keys->shape(2)})
            throw std::runtime_error("invalid QSA index tail");
        writer.scalar<std::int32_t>(ratio);
        writer.scalar<std::int32_t>(exact ? layer.index_start : boundary);
        write_optional_tensor(writer, exact ? layer.index_keys : std::nullopt);
        const int pool_start = static_cast<int>(start) / ratio;
        const int pool_end = boundary / ratio;
        if (pool_end > pool_start && layer.offloaded_pooled_keys) {
            writer.scalar<std::uint8_t>(1);
            writer.offloaded_index_tensor(*layer.offloaded_pooled_keys, pool_start, pool_end - pool_start);
        } else if (pool_end > pool_start && layer.pooled_keys) {
            write_optional_tensor(writer, mlx::core::slice(*layer.pooled_keys, Shape{0, pool_start, 0},
                Shape{1, pool_end, layer.pooled_keys->shape(2)}));
        } else {
            if (pool_end > pool_start) throw std::runtime_error("missing QSA block keys");
            write_optional_tensor(writer, std::nullopt);
        }
    } else {
        MlxQwen35LinearAttentionCacheSnapshot state{*layer.convolution, *layer.recurrent,
            layer.position, layer.batch};
        if (exact) write_recurrent_layer(writer, state, boundary, kGdnLayer);
        else write_recurrent_unavailable_layer(writer, state, boundary, kGdnUnavailableLayer);
    }
    write_optional_tensor(writer, exact ? layer.ple_convolution : std::nullopt);
    writer.scalar<std::uint32_t>(exact ? static_cast<std::uint32_t>(layer.ple_context.size()) : 0);
    if (exact) for (const auto token : layer.ple_context) writer.scalar<std::int64_t>(token);
}

DecodedLayer read_layer(Reader& reader) {
    DecodedLayer layer;
    layer.kind = reader.scalar<std::uint8_t>("layer kind");
    const auto dtype_value = reader.scalar<std::uint8_t>("layer dtype");
    const auto flags = reader.scalar<std::uint16_t>("layer flags");
    layer.exact = (flags & 1) != 0;
    if (flags & 2) {
        layer.quantization.bits = (flags >> 8) / 2.0;
        if (!layer.quantization.enabled() || !mlx_kv_bits_valid(layer.quantization.bits))
            throw std::runtime_error("invalid TurboQuant KV checkpoint bit width");
    } else if (flags & 0xff00) throw std::runtime_error("invalid KV checkpoint quantization flags");
    layer.batch = reader.scalar<std::int32_t>("batch");
    layer.heads = reader.scalar<std::int32_t>("heads");
    layer.maximum_sequence = reader.scalar<std::int32_t>("maximum sequence");
    layer.head_dimension = reader.scalar<std::int32_t>("head dimension");
    layer.capacity = reader.scalar<std::int32_t>("capacity");
    layer.position = reader.scalar<std::int32_t>("position");
    const bool unavailable = layer.kind == kRecurrentUnavailableLayer || layer.kind == kGdnUnavailableLayer;
    if (!unavailable) { layer.first = reader.tensor(); layer.second = reader.tensor(); }
    if (layer.kind == kKvLayer || layer.kind == kQsaLayer) {
        layer.dtype = layer.first.dtype;
        if (layer.quantization.enabled() && layer.dtype != mlx::core::uint32)
            throw std::runtime_error("invalid TurboQuant KV checkpoint dtype");
        if (dtype_value > static_cast<std::uint8_t>(Dtype::Val::complex64) ||
            layer.dtype.val() != static_cast<Dtype::Val>(dtype_value))
            throw std::runtime_error("MLX KV layer dtype mismatch");
    } else if (unavailable || layer.kind == kRecurrentLayer || layer.kind == kGdnLayer) {
        if (dtype_value != 0 || layer.batch != 1 || layer.heads != 0 || layer.maximum_sequence != 0 ||
            layer.head_dimension != 0 || layer.capacity != 0 || layer.position <= 0 ||
            (!unavailable && (layer.first.dtype != mlx::core::float32 || layer.second.dtype != mlx::core::float32)))
            throw std::runtime_error("invalid recurrent cache checkpoint");
    } else throw std::runtime_error("unknown MLX cache layer kind");
    if (layer.kind == kQsaLayer) {
        layer.index_ratio = reader.scalar<std::int32_t>("index ratio");
        layer.index_start = reader.scalar<std::int32_t>("index tail start");
        if (layer.index_ratio <= 0 || layer.index_start < 0 || layer.index_start > layer.position)
            throw std::runtime_error("invalid QSA index tail geometry");
        if (reader.scalar<std::uint8_t>("index tail presence")) layer.index = reader.tensor();
        if (reader.scalar<std::uint8_t>("pooled presence")) layer.pooled = reader.tensor();
    }
    if (layer.kind == kQsaLayer || layer.kind == kGdnLayer || layer.kind == kGdnUnavailableLayer) {
        if (reader.scalar<std::uint8_t>("PLE presence")) layer.ple = reader.tensor();
        const auto count = reader.scalar<std::uint32_t>("PLE context count");
        if (count > 65536) throw std::runtime_error("invalid PLE context count");
        for (std::uint32_t i = 0; i < count; ++i)
            layer.ple_context.push_back(reader.scalar<std::int64_t>("PLE token"));
    }
    return layer;
}

DecodedBlock read_block(const std::vector<std::uint8_t>& payload) {
    Reader reader(payload);
    std::array<std::uint8_t, 8> magic{};
    reader.raw(magic.data(), magic.size(), "magic");
    const auto version = reader.scalar<std::uint32_t>("version");
    if (magic != kMagic || version < 1 || version > kVersion) {
        throw std::runtime_error("unsupported MLX cache payload");
    }
    DecodedBlock block;
    block.runtime = reader.scalar<std::uint32_t>("runtime");
    if (block.runtime == kQwen4Runtime && version < 3)
        throw std::runtime_error("obsolete Flash-Next index history cache");
    block.start = reader.scalar<std::uint32_t>("start");
    block.count = reader.scalar<std::uint32_t>("count");
    const auto layer_count = reader.scalar<std::uint32_t>("layer count");
    if (block.count == 0 || layer_count == 0 || layer_count > 4096) {
        throw std::runtime_error("invalid MLX cache block geometry");
    }
    block.layers.reserve(layer_count);
    for (std::uint32_t index = 0; index < layer_count; ++index) {
        block.layers.push_back(read_layer(reader));
    }
    if (version >= 2) {
        if (reader.scalar<std::uint8_t>("MTP hidden presence")) block.last_hidden = reader.tensor();
        const auto count = reader.scalar<std::uint32_t>("MTP layer count");
        if (count > 4096 || (count > 0 && !block.last_hidden))
            throw std::runtime_error("invalid MTP checkpoint");
        for (std::uint32_t i = 0; i < count; ++i) block.mtp_layers.push_back(read_layer(reader));
    }
    reader.expect_end();
    return block;
}

std::vector<DecodedBlock> decode_blocks(
    const std::vector<MlxPagedPayload>& payloads,
    std::uint32_t expected_runtime,
    std::size_t token_count,
    std::size_t block_size) {
    if (payloads.empty() || block_size == 0 || token_count == 0 ||
        (token_count + block_size - 1) / block_size != payloads.size()) {
        throw std::runtime_error("MLX cache block chain length mismatch");
    }
    std::vector<DecodedBlock> blocks;
    blocks.reserve(payloads.size());
    std::size_t expected_start = 0;
    std::size_t expected_layers = 0;
    for (const auto& payload : payloads) {
        if (!payload) {
            throw std::runtime_error("MLX cache block payload is null");
        }
        auto block = read_block(*payload);
        if (block.runtime != expected_runtime || block.start != expected_start ||
            block.count != std::min(block_size, token_count - expected_start) ||
            (expected_layers != 0 && block.layers.size() != expected_layers)) {
            throw std::runtime_error("incompatible MLX cache block chain");
        }
        const auto boundary = static_cast<std::uint64_t>(block.start) +
            static_cast<std::uint64_t>(block.count);
        if (boundary >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int32_t>::max()) ||
            std::any_of(
                block.layers.begin(),
                block.layers.end(),
                [boundary](const DecodedLayer& layer) {
                    return layer.position != static_cast<int>(boundary);
                })) {
            throw std::runtime_error("MLX cache layer boundary mismatch");
        }
        expected_start += block.count;
        expected_layers = block.layers.size();
        blocks.push_back(std::move(block));
    }
    return blocks;
}

MlxKvCacheSnapshot rebuild_kv(
    const std::vector<DecodedBlock>& blocks,
    std::size_t layer_index,
    std::size_t token_count) {
    if (token_count > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("MLX cache token count is too large");
    }
    const auto& final = blocks.back().layers.at(layer_index);
    if (final.batch <= 0 || final.heads <= 0 ||
        final.head_dimension <= 0) {
        throw std::runtime_error("invalid MLX KV cache topology");
    }
    const auto expected_shape = [&](std::size_t count, bool key) {
        return Shape{
            final.batch,
            final.heads,
            static_cast<int>(count),
            final.quantization.enabled() ? mlx_kv_packed_width(final.head_dimension,
                key ? final.quantization.key_bits() : final.quantization.value_bits()) : final.head_dimension,
        };
    };
    for (const auto& block : blocks) {
        const auto& layer = block.layers.at(layer_index);
        if ((layer.kind != kKvLayer && layer.kind != kQsaLayer) || layer.batch != final.batch ||
            layer.heads != final.heads ||
            layer.maximum_sequence != final.maximum_sequence ||
            layer.head_dimension != final.head_dimension ||
            layer.dtype != final.dtype ||
            layer.quantization != final.quantization ||
            layer.first.dtype != final.dtype ||
            layer.second.dtype != final.dtype ||
            layer.first.shape != expected_shape(block.count, true) ||
            layer.second.shape != expected_shape(block.count, false) ||
            layer.first.data == nullptr || layer.second.data == nullptr ||
            static_cast<std::size_t>(block.start) + block.count > token_count) {
            throw std::runtime_error("inconsistent MLX KV cache block topology");
        }
    }

    const auto element_bytes = final.dtype.size();
    const auto batch = static_cast<std::size_t>(final.batch);
    const auto heads = static_cast<std::size_t>(final.heads);
    const auto dimension = static_cast<std::size_t>(std::max(expected_shape(1, true)[3], expected_shape(1, false)[3]));
    if (batch > std::numeric_limits<std::size_t>::max() / heads ||
        token_count > std::numeric_limits<std::size_t>::max() / dimension ||
        token_count * dimension >
            std::numeric_limits<std::size_t>::max() / element_bytes) {
        throw std::runtime_error("MLX KV cache tensor size overflow");
    }
    const auto lanes = batch * heads;
    const auto target_lane_bytes = token_count *
        dimension * element_bytes;
    if (target_lane_bytes != 0 &&
        lanes > std::numeric_limits<std::size_t>::max() / target_lane_bytes) {
        throw std::runtime_error("MLX KV cache allocation size overflow");
    }
    auto rebuild = [&](bool key) {
        const auto target_shape = expected_shape(token_count, key);
        const auto dimension = static_cast<std::size_t>(target_shape[3]);
        const auto target_lane_bytes = token_count * dimension * element_bytes;
        const auto total_bytes = lanes * target_lane_bytes;
        MlxResidentBudgetScope::reserve(total_bytes);
        auto result = array(
            mlx::core::allocator::malloc(total_bytes),
            target_shape,
            final.dtype);
        auto* destination = result.data<std::uint8_t>();
        for (const auto& block : blocks) {
            const auto& tensor = key
                ? block.layers.at(layer_index).first
                : block.layers.at(layer_index).second;
            const auto block_lane_bytes =
                static_cast<std::size_t>(block.count) *
                dimension * element_bytes;
            if (tensor.bytes != lanes * block_lane_bytes) {
                throw std::runtime_error("invalid MLX KV cache tensor size");
            }
            const auto destination_offset =
                static_cast<std::size_t>(block.start) *
                dimension * element_bytes;
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                std::memcpy(
                    destination + lane * target_lane_bytes +
                        destination_offset,
                    tensor.data + lane * block_lane_bytes,
                    block_lane_bytes);
            }
        }
        return result;
    };
    auto key = rebuild(true);
    auto value = rebuild(false);
    const int position = static_cast<int>(token_count);
    return MlxKvCacheSnapshot{
        final.batch,
        final.heads,
        final.maximum_sequence,
        final.head_dimension,
        std::max(final.capacity, position),
        position,
        final.dtype,
        std::move(key),
        std::move(value),
        final.quantization,
    };
}

array copy_tensor(const SerializedTensor& source) {
    if (source.data == nullptr || source.bytes == 0) throw std::runtime_error("empty cache tensor");
    MlxResidentBudgetScope::reserve(source.bytes);
    auto result = array(mlx::core::allocator::malloc(source.bytes), source.shape, source.dtype);
    std::memcpy(result.data<std::uint8_t>(), source.data, source.bytes);
    return result;
}

MlxQwen4LayerCacheSnapshot rebuild_flash(const std::vector<DecodedBlock>& blocks,
    std::size_t index, std::size_t count) {
    const auto& final = blocks.back().layers.at(index);
    MlxQwen4LayerCacheSnapshot state;
    state.batch = final.batch;
    state.position = static_cast<int>(count);
    if (final.kind == kQsaLayer) {
        if (auto store = mlx_qsa_kv_offload_store()) {
            if (final.batch != 1 || final.heads <= 0 || final.head_dimension <= 0 ||
                (final.quantization.enabled() ? final.dtype != mlx::core::uint32 :
                    (final.dtype != mlx::core::float16 && final.dtype != mlx::core::bfloat16)))
                throw std::runtime_error("invalid offloaded QSA checkpoint geometry");
            const int key_width = final.quantization.enabled() ? mlx_kv_packed_width(final.head_dimension, final.quantization.key_bits()) : final.head_dimension;
            const int value_width = final.quantization.enabled() ? mlx_kv_packed_width(final.head_dimension, final.quantization.value_bits()) : final.head_dimension;
            const auto key_lane = static_cast<std::size_t>(key_width) * final.dtype.size();
            const auto value_lane = static_cast<std::size_t>(value_width) * final.dtype.size();
            const auto key_bytes = key_lane * final.heads, value_bytes = value_lane * final.heads;
            QsaKvSequence rows(store, key_bytes + value_bytes, store->config().microblock_rows, 1);
            std::vector<std::uint8_t> row(key_bytes + value_bytes);
            for (const auto& block : blocks) {
                const auto& layer = block.layers.at(index);
                if (block.start != rows.position() || block.start + block.count > count ||
                    layer.kind != kQsaLayer || layer.heads != final.heads ||
                    layer.head_dimension != final.head_dimension || layer.maximum_sequence != final.maximum_sequence ||
                    layer.quantization != final.quantization ||
                    layer.first.shape != Shape{1, final.heads, static_cast<int>(block.count), key_width} ||
                    layer.second.shape != Shape{1, final.heads, static_cast<int>(block.count), value_width} ||
                    layer.first.dtype != final.dtype || layer.second.dtype != final.dtype ||
                    layer.first.bytes != block.count * key_bytes || layer.second.bytes != block.count * value_bytes ||
                    !layer.first.data || !layer.second.data)
                    throw std::runtime_error("offloaded QSA checkpoint block topology mismatch");
                for (std::size_t token = 0; token < block.count; ++token) {
                    for (int head = 0; head < final.heads; ++head) {
                        std::memcpy(row.data() + head * key_lane, layer.first.data + (head * block.count + token) * key_lane, key_lane);
                        std::memcpy(row.data() + key_bytes + head * value_lane, layer.second.data + (head * block.count + token) * value_lane, value_lane);
                    }
                    rows.append(row.data(), 1);
                }
            }
            if (rows.position() != count) throw std::runtime_error("incomplete offloaded QSA checkpoint");
            rows.seal_tail();
            state.offloaded_kv = MlxQsaKvSnapshot{rows.snapshot(), final.heads, final.head_dimension,
                final.maximum_sequence, final.dtype, final.quantization};
        } else state.kv = rebuild_kv(blocks, index, count);
        if (!final.index || final.index->shape.size() != 3)
            throw std::runtime_error("missing QSA index checkpoint");
        const int width = final.index->shape[2];
        const int ratio = final.index_ratio;
        if (width <= 0 || ratio <= 0 || final.index_start < 0 || final.index_start % ratio ||
            final.index_start > static_cast<int>(count) ||
            final.index->shape != Shape{1, static_cast<int>(count) - final.index_start, width} ||
            final.index->dtype != mlx::core::float16 ||
            count - final.index_start > static_cast<std::size_t>(ratio + kMlxMtpEngineMaximumDraftDepth + 1))
            throw std::runtime_error("QSA index tail checkpoint mismatch");
        state.index_start = final.index_start;
        state.index_ratio = ratio;
        state.index_keys = copy_tensor(*final.index);
        const auto pool_count = count / ratio;
        if (pool_count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(width) / 4)
            throw std::runtime_error("QSA block index checkpoint size overflow");
        std::optional<array> pooled;
        std::unique_ptr<QsaKvSequence> pool_rows;
        if (auto store = mlx_qsa_kv_offload_store())
            pool_rows = std::make_unique<QsaKvSequence>(store, width * 4, 16, 2);
        else if (pool_count) {
            MlxResidentBudgetScope::reserve(pool_count * width * 4);
            pooled = array(mlx::core::allocator::malloc(pool_count * width * 4),
                Shape{1, static_cast<int>(pool_count), width}, mlx::core::float32);
        }
        for (const auto& block : blocks) {
            const auto& layer = block.layers.at(index);
            const auto begin = block.start / ratio;
            const auto rows = (block.start + block.count) / ratio - begin;
            if (layer.index_ratio != ratio || (rows && (!layer.pooled ||
                layer.pooled->dtype != mlx::core::float32 ||
                layer.pooled->shape != Shape{1, static_cast<int>(rows), width})))
                throw std::runtime_error("QSA pooled index block topology mismatch");
            if (!rows) continue;
            if (pool_rows) {
                if (pool_rows->position() != begin) throw std::runtime_error("QSA block index chain mismatch");
                pool_rows->append(layer.pooled->data, rows);
            } else std::memcpy(pooled->data<std::uint8_t>() + begin * width * 4,
                layer.pooled->data, layer.pooled->bytes);
        }
        if (pool_rows && pool_count) {
            pool_rows->seal_tail();
            state.offloaded_pooled_keys = MlxQsaIndexSnapshot{pool_rows->snapshot(), width, mlx::core::float32};
        } else state.pooled_keys = std::move(pooled);
    } else if (final.kind == kGdnLayer) {
        state.convolution = copy_tensor(final.first);
        state.recurrent = copy_tensor(final.second);
    } else throw std::runtime_error("missing exact GDN checkpoint");
    if (final.ple) state.ple_convolution = copy_tensor(*final.ple);
    state.ple_context = final.ple_context;
    return state;
}

MlxQwen35LinearAttentionCacheSnapshot rebuild_recurrent(
    const std::vector<DecodedBlock>& blocks,
    std::size_t layer_index,
    std::size_t token_count) {
    const auto& final = blocks.back().layers.at(layer_index);
    if (final.kind != kRecurrentLayer || final.batch != 1 ||
        final.position != static_cast<int>(token_count) ||
        final.first.data == nullptr || final.second.data == nullptr) {
        throw std::runtime_error("invalid recurrent cache boundary block");
    }
    const auto copy = [](const SerializedTensor& source) {
        MlxResidentBudgetScope::reserve(source.bytes);
        auto result = array(
            mlx::core::allocator::malloc(source.bytes),
            source.shape,
            source.dtype);
        std::memcpy(result.data<std::uint8_t>(), source.data, source.bytes);
        return result;
    };
    return MlxQwen35LinearAttentionCacheSnapshot{
        copy(final.first),
        copy(final.second),
        static_cast<int>(token_count),
        final.batch,
    };
}

template <typename State, typename LayerWriter>
std::vector<MlxPagedPayload> encode_state(
    const State& state,
    std::size_t block_size,
    std::size_t first_block,
    std::size_t maximum_blocks,
    std::uint32_t runtime,
    LayerWriter write_layer,
    bool allow_tail = false) {
    if (block_size == 0 || state.cache_batch != 1 ||
        state.cache_position <= 0 ||
        state.tokens.size() != static_cast<std::size_t>(state.cache_position) ||
        state.layers.empty()) {
        throw std::runtime_error("invalid MLX session state for paging");
    }
    const auto full_blocks = allow_tail ? (state.tokens.size() + block_size - 1) / block_size
        : state.tokens.size() / block_size;
    if (first_block > full_blocks) {
        throw std::runtime_error("invalid MLX cache first block");
    }
    const auto end_block = maximum_blocks > full_blocks - first_block
        ? full_blocks
        : first_block + maximum_blocks;
    std::vector<MlxPagedPayload> result;
    result.reserve(end_block - first_block);
    for (std::size_t block = first_block; block < end_block; ++block) {
        const auto start = block * block_size;
        const auto count = std::min(block_size, state.tokens.size() - start);
        const bool exact_boundary = start + count == state.tokens.size();
        Writer writer;
        write_header(
            writer,
            runtime,
            static_cast<std::uint32_t>(start),
            static_cast<std::uint32_t>(count),
            state.layers.size());
        for (const auto& layer : state.layers) {
            write_layer(
                writer,
                layer,
                start,
                count,
                exact_boundary);
        }
        if constexpr (requires { state.mtp_layers; state.last_hidden; }) {
            write_optional_tensor(writer, exact_boundary ? state.last_hidden : std::nullopt);
            writer.scalar<std::uint32_t>(exact_boundary && state.last_hidden
                ? static_cast<std::uint32_t>(state.mtp_layers.size()) : 0);
            if (exact_boundary && state.last_hidden) for (const auto& layer : state.mtp_layers) {
                if constexpr (std::is_same_v<std::decay_t<decltype(layer)>, MlxKvCacheSnapshot>)
                    write_kv_layer(writer, layer, 0, layer.position);
                else write_flash_layer(writer, layer, 0, layer.position, true);
            }
        } else {
            writer.scalar<std::uint8_t>(0);
            writer.scalar<std::uint32_t>(0);
        }
        result.push_back(std::move(writer).finish());
    }
    return result;
}

} // namespace

std::vector<MlxPagedPayload>
MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::encode(
    const MlxMiniCPMO45TextSessionState& state,
    std::size_t block_size,
    std::size_t first_block) {
    return encode_state(
        state,
        block_size,
        first_block,
        std::numeric_limits<std::size_t>::max(),
        kMiniRuntime,
        [](Writer& writer, const MlxKvCacheSnapshot& layer,
           std::size_t start, std::size_t count, bool) {
            write_kv_layer(
                writer,
                layer,
                static_cast<int>(start),
                static_cast<int>(count));
        });
}

MlxPagedPayload
MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::encode_block(
    const MlxMiniCPMO45TextSessionState& state,
    std::size_t block_size,
    std::size_t block_index) {
    auto result = encode_state(
        state,
        block_size,
        block_index,
        1,
        kMiniRuntime,
        [](Writer& writer, const MlxKvCacheSnapshot& layer,
           std::size_t start, std::size_t count, bool) {
            write_kv_layer(
                writer,
                layer,
                static_cast<int>(start),
                static_cast<int>(count));
        });
    if (result.size() != 1) {
        throw std::runtime_error("invalid MiniCPM cache block index");
    }
    return std::move(result.front());
}

MlxMiniCPMO45TextSessionState
MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
    const std::vector<MlxPagedPayload>& payloads,
    const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    auto blocks = decode_blocks(
        payloads, kMiniRuntime, tokens.size(), block_size);
    MlxMiniCPMO45TextSessionState state;
    state.tokens = tokens;
    state.cache_position = static_cast<int>(tokens.size());
    state.cache_batch = 1;
    state.layers.reserve(blocks.front().layers.size());
    for (std::size_t index = 0; index < blocks.front().layers.size(); ++index) {
        auto layer = rebuild_kv(blocks, index, tokens.size());
        state.bytes += layer.nbytes();
        state.layers.push_back(std::move(layer));
    }
    return state;
}

std::vector<MlxPagedPayload>
MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode(
    const MlxQwen35TextSessionState& state,
    std::size_t block_size,
    std::size_t first_block) {
    return encode_state(
        state,
        block_size,
        first_block,
        std::numeric_limits<std::size_t>::max(),
        kQwen35Runtime,
        [](Writer& writer, const MlxQwen35LayerCacheSnapshot& layer,
           std::size_t start, std::size_t count, bool exact_boundary) {
            std::visit(
                [&](const auto& snapshot) {
                    using Snapshot = std::decay_t<decltype(snapshot)>;
                    if constexpr (std::is_same_v<Snapshot, MlxKvCacheSnapshot>) {
                        write_kv_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start),
                            static_cast<int>(count));
                    } else if (exact_boundary) {
                        write_recurrent_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start + count));
                    } else {
                        write_recurrent_unavailable_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start + count));
                    }
                },
                layer);
        }, true);
}

MlxPagedPayload
MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode_block(
    const MlxQwen35TextSessionState& state,
    std::size_t block_size,
    std::size_t block_index) {
    auto result = encode_state(
        state,
        block_size,
        block_index,
        1,
        kQwen35Runtime,
        [](Writer& writer, const MlxQwen35LayerCacheSnapshot& layer,
           std::size_t start, std::size_t count, bool exact_boundary) {
            std::visit(
                [&](const auto& snapshot) {
                    using Snapshot = std::decay_t<decltype(snapshot)>;
                    if constexpr (std::is_same_v<
                                      Snapshot, MlxKvCacheSnapshot>) {
                        write_kv_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start),
                            static_cast<int>(count));
                    } else if (exact_boundary) {
                        write_recurrent_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start + count));
                    } else {
                        write_recurrent_unavailable_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start + count));
                    }
                },
                layer);
        }, true);
    if (result.size() != 1) {
        throw std::runtime_error("invalid Qwen3.5 cache block index");
    }
    return std::move(result.front());
}

MlxQwen35TextSessionState
MlxPagedSessionCodec<MlxQwen35TextSessionState>::decode(
    const std::vector<MlxPagedPayload>& payloads,
    const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    auto blocks = decode_blocks(
        payloads, kQwen35Runtime, tokens.size(), block_size);
    MlxQwen35TextSessionState state;
    state.tokens = tokens;
    state.cache_position = static_cast<int>(tokens.size());
    state.cache_batch = 1;
    state.layers.reserve(blocks.front().layers.size());
    for (std::size_t index = 0; index < blocks.front().layers.size(); ++index) {
        if (blocks.front().layers[index].kind == kKvLayer) {
            auto layer = rebuild_kv(blocks, index, tokens.size());
            state.bytes += layer.nbytes();
            state.layers.emplace_back(std::move(layer));
        } else {
            auto layer = rebuild_recurrent(blocks, index, tokens.size());
            state.bytes += layer.nbytes();
            state.layers.emplace_back(std::move(layer));
        }
    }
    const auto& final = blocks.back();
    if (final.last_hidden) {
        state.last_hidden = copy_tensor(*final.last_hidden);
        state.bytes += state.last_hidden->nbytes();
        for (const auto& layer : final.mtp_layers) {
            if (layer.position != state.cache_position - 1 || layer.kind != kKvLayer)
                throw std::runtime_error("MTP checkpoint position mismatch");
            DecodedBlock head;
            head.count = static_cast<std::uint32_t>(layer.position);
            head.layers.push_back(layer);
            auto snapshot = rebuild_kv({head}, 0, layer.position);
            state.bytes += snapshot.nbytes();
            state.mtp_layers.push_back(std::move(snapshot));
        }
    }
    return state;
}

std::size_t MlxPagedSessionCodec<MlxQwen35TextSessionState>::token_count(const MlxPagedPayload& payload) {
    if (!payload) throw std::runtime_error("null cache payload");
    const auto block = read_block(*payload);
    return block.start + block.count;
}

bool MlxPagedSessionCodec<MlxQwen35TextSessionState>::has_mtp(const MlxPagedPayload& payload) {
    if (!payload) throw std::runtime_error("null cache payload");
    const auto block = read_block(*payload);
    if (block.runtime != kQwen35Runtime) throw std::runtime_error("incompatible Qwen3.5 cache block");
    return block.last_hidden && (block.start + block.count == 1 || !block.mtp_layers.empty());
}

bool MlxPagedSessionCodec<MlxQwen4TextSessionState>::has_mtp(const MlxPagedPayload& payload) {
    if (!payload) throw std::runtime_error("null cache payload");
    const auto block = read_block(*payload);
    if (block.runtime != kQwen4Runtime) throw std::runtime_error("incompatible Flash-Next cache block");
    return block.last_hidden && (block.start + block.count == 1 || !block.mtp_layers.empty());
}

MlxPagedPayload MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(
    const MlxQwen4TextSessionState& state, std::size_t block_size, std::size_t index) {
    auto blocks = encode_state(state, block_size, index, 1, kQwen4Runtime, write_flash_layer, true);
    if (blocks.size() != 1) throw std::runtime_error("invalid Flash-Next cache block index");
    return std::move(blocks.front());
}

MlxQwen4TextSessionState MlxPagedSessionCodec<MlxQwen4TextSessionState>::decode(
    const std::vector<MlxPagedPayload>& payloads, const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    const auto blocks = decode_blocks(payloads, kQwen4Runtime, tokens.size(), block_size);
    MlxQwen4TextSessionState state;
    state.tokens = tokens;
    state.cache_position = static_cast<int>(tokens.size());
    state.cache_batch = 1;
    for (std::size_t i = 0; i < blocks.front().layers.size(); ++i)
        state.layers.push_back(rebuild_flash(blocks, i, tokens.size()));
    const auto& final = blocks.back();
    if (final.last_hidden) {
        state.last_hidden = copy_tensor(*final.last_hidden);
        for (const auto& layer : final.mtp_layers) {
            if (layer.position != state.cache_position - 1 || layer.kind != kQsaLayer)
                throw std::runtime_error("Flash-Next MTP checkpoint position mismatch");
            DecodedBlock head;
            head.count = static_cast<std::uint32_t>(layer.position);
            head.layers.push_back(layer);
            state.mtp_layers.push_back(rebuild_flash({head}, 0, layer.position));
        }
    }
    const auto bytes = [&](const MlxQwen4LayerCacheSnapshot& layer) {
        if (layer.kv) state.bytes += layer.kv->nbytes();
        if (layer.offloaded_kv) state.bytes += layer.offloaded_kv->rows.tail.size() +
            layer.offloaded_kv->rows.blocks.size() * sizeof(QsaKvStore::BlockPtr);
        if (layer.offloaded_pooled_keys) state.bytes += layer.offloaded_pooled_keys->rows.tail.size() +
            layer.offloaded_pooled_keys->rows.blocks.size() * sizeof(QsaKvStore::BlockPtr);
        for (const auto* value : {&layer.index_keys, &layer.pooled_keys, &layer.convolution,
            &layer.recurrent, &layer.ple_convolution}) if (*value) state.bytes += (*value)->nbytes();
        state.bytes += layer.ple_context.size() * sizeof(std::int64_t);
    };
    for (const auto& layer : state.layers) bytes(layer);
    for (const auto& layer : state.mtp_layers) bytes(layer);
    if (state.last_hidden) state.bytes += state.last_hidden->nbytes();
    return state;
}

bool MlxPagedSessionCodec<MlxQwen4TextSessionState>::has_exact_boundary(const MlxPagedPayload& payload) {
    if (!payload) throw std::runtime_error("null cache payload");
    const auto block = read_block(*payload);
    if (block.runtime != kQwen4Runtime) throw std::runtime_error("incompatible Flash-Next cache block");
    const auto boundary = static_cast<std::uint64_t>(block.start) + block.count;
    if (boundary > std::numeric_limits<std::int32_t>::max() ||
        std::any_of(block.layers.begin(), block.layers.end(), [boundary](const DecodedLayer& layer) {
            return layer.position != static_cast<int>(boundary);
        })) throw std::runtime_error("Flash-Next cache checkpoint boundary mismatch");
    return std::none_of(block.layers.begin(), block.layers.end(), [](const DecodedLayer& layer) {
        return layer.kind == kGdnUnavailableLayer || (layer.kind == kQsaLayer && !layer.exact);
    });
}

std::size_t MlxPagedSessionCodec<MlxQwen4TextSessionState>::decodable_blocks(
    const std::vector<MlxPagedPayload>& payloads) {
    for (std::size_t count = payloads.size(); count > 0; --count)
        if (has_exact_boundary(payloads[count - 1])) return count;
    return 0;
}

std::size_t MlxPagedSessionCodec<MlxQwen4TextSessionState>::token_count(const MlxPagedPayload& payload) {
    if (!payload) throw std::runtime_error("null cache payload");
    const auto block = read_block(*payload);
    return block.start + block.count;
}

std::size_t
MlxPagedSessionCodec<MlxQwen35TextSessionState>::decodable_blocks(
    const std::vector<MlxPagedPayload>& payloads) {
    for (std::size_t count = payloads.size(); count > 0; --count) {
        if (has_exact_boundary(payloads[count - 1])) return count;
    }
    return 0;
}

bool MlxPagedSessionCodec<MlxQwen35TextSessionState>::has_exact_boundary(
    const MlxPagedPayload& payload) {
    if (!payload) {
        throw std::runtime_error("MLX cache block payload is null");
    }
    const auto block = read_block(*payload);
    if (block.runtime != kQwen35Runtime) {
        throw std::runtime_error("incompatible Qwen3.5 cache block");
    }
    const auto boundary = static_cast<std::uint64_t>(block.start) +
        static_cast<std::uint64_t>(block.count);
    if (boundary >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max()) ||
        std::any_of(
            block.layers.begin(),
            block.layers.end(),
            [boundary](const DecodedLayer& layer) {
                return layer.position != static_cast<int>(boundary);
            })) {
        throw std::runtime_error("Qwen3.5 cache checkpoint boundary mismatch");
    }
    return std::none_of(
        block.layers.begin(),
        block.layers.end(),
        [](const DecodedLayer& layer) {
            return layer.kind == kRecurrentUnavailableLayer;
        });
}

} // namespace mfq::metal
