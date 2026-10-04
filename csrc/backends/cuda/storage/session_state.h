#pragma once

#include "../native/tensor_backend.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

struct Dsv4PoolState;
struct FullBlock;

struct FullBlockSessionState {
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
    int64_t capacity = 0;
    bool ring = false;
};

class CudaSessionStateError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class TextSessionStateKind : int {
    Unsupported,
    FullAttention,
    HybridAttention,
    DeepseekV4,
    GlmDsa,
};

struct Dsv4PoolSessionState {
    int64_t ratio = 0;
    int64_t head_dim = 0;
    int64_t cache_quant_mode = 0;
    int64_t capacity = 0;
    bool overlap = false;
    mfq_tensor_backend::Tensor state_kv;
    mfq_tensor_backend::Tensor state_gate;
    mfq_tensor_backend::Tensor previous_kv;
    mfq_tensor_backend::Tensor previous_gate;
    mfq_tensor_backend::Tensor pool;
};

struct Dsv4BlockSessionState {
    mfq_tensor_backend::Tensor local_cache;
    Dsv4PoolSessionState compressor;
    Dsv4PoolSessionState indexer_compressor;
};

struct GlmDsaBlockSessionState {
    mfq_tensor_backend::Tensor kv_cache;
    mfq_tensor_backend::Tensor index_cache;
    int64_t kv_capacity = 0;
    int64_t index_capacity = 0;
    bool full_indexer = false;
};

enum class HybridBlockSessionStateKind {
    FullAttention,
    Recurrent,
};

struct HybridBlockSessionState {
    HybridBlockSessionStateKind kind =
        HybridBlockSessionStateKind::FullAttention;
    FullBlockSessionState full_attention;
    mfq_tensor_backend::Tensor convolution_state;
    mfq_tensor_backend::Tensor recurrent_state;
};

struct MtpSessionState {
    std::vector<FullBlockSessionState> blocks;
    mfq_tensor_backend::Tensor last_target_hidden;
    int64_t cache_pos = 0;
    size_t bytes = 0;
};

using TextSessionPayload = std::variant<
    std::monostate,
    std::vector<FullBlockSessionState>,
    std::vector<HybridBlockSessionState>,
    std::vector<Dsv4BlockSessionState>,
    std::vector<GlmDsaBlockSessionState>>;

struct TextSessionState {
    std::vector<int64_t> tokens;
    std::string input_key;
    TextSessionPayload payload;
    std::optional<MtpSessionState> mtp;
    int64_t cache_pos = 0;
    int64_t decode_position_delta = 0;
    size_t bytes = 0;

    TextSessionStateKind kind() const noexcept {
        return static_cast<TextSessionStateKind>(payload.index());
    }
};

namespace mfq::cuda {
template <typename Model> struct CudaCausalOps;
template <typename Model> struct FullAttentionSessionCodec {
    using CausalModel = typename Model::template CausalModel<CudaCausalOps<Model>>;
    static TextSessionStateKind kind(const CausalModel &model);
    static bool supports_paged(const CausalModel &model);
    static TextSessionState capture(const CausalModel &model, const std::vector<int64_t> &tokens);
    static void restore(CausalModel &model, const TextSessionState &state);
};
} // namespace mfq::cuda

using CudaPagedPayload =
    std::shared_ptr<const std::vector<uint8_t>>;

class CudaPagedWriter {
public:
    template <typename T>
    void scalar(T value) {
        using Unsigned = std::make_unsigned_t<T>;
        const auto converted = static_cast<Unsigned>(value);
        for (size_t index = 0; index < sizeof(T); ++index) {
            bytes_.push_back(static_cast<uint8_t>(
                converted >> (index * 8)));
        }
    }

    void raw(const void* data, size_t size);
    void tensor(const mfq_tensor_backend::Tensor& source);
    CudaPagedPayload finish() &&;

private:
    std::vector<uint8_t> bytes_;
};

class CudaPagedReader {
public:
    explicit CudaPagedReader(const std::vector<uint8_t>& bytes)
        : cursor_(bytes.data()), end_(bytes.data() + bytes.size()) {}

    template <typename T>
    T scalar(const char* name) {
        if (remaining() < sizeof(T)) {
            throw CudaSessionStateError(
                std::string("truncated CUDA paged cache ") + name);
        }
        using Unsigned = std::make_unsigned_t<T>;
        Unsigned value = 0;
        for (size_t index = 0; index < sizeof(T); ++index) {
            value |= static_cast<Unsigned>(cursor_[index]) << (index * 8);
        }
        cursor_ += sizeof(T);
        return static_cast<T>(value);
    }

    void raw(void* destination, size_t size, const char* name);
    std::pair<mfq_tensor_backend::Tensor, int> tensor();
    void expect_end() const;

private:
    size_t remaining() const;

    const uint8_t* cursor_;
    const uint8_t* end_;
};

struct CudaDecodedPagedLayer {
    int64_t capacity = 0;
    int device = -1;
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
};

struct CudaDecodedPagedBlock {
    uint32_t start = 0;
    uint32_t count = 0;
    std::vector<CudaDecodedPagedLayer> layers;
};

std::vector<CudaPagedPayload> encode_cuda_paged_session(
    const TextSessionState& state,
    size_t block_size,
    size_t first_block,
    size_t maximum_blocks = std::numeric_limits<size_t>::max());

CudaPagedPayload encode_cuda_paged_block(
    const TextSessionState& state,
    size_t block_size,
    size_t block_index);

CudaDecodedPagedBlock decode_cuda_paged_block(
    const std::vector<uint8_t>& payload);

TextSessionState decode_cuda_paged_session(
    const std::vector<CudaPagedPayload>& payloads,
    const std::vector<int64_t>& tokens,
    size_t block_size);

size_t session_tensor_bytes(const mfq_tensor_backend::Tensor& tensor);

bool session_tensor_layout_matches(
    const mfq_tensor_backend::Tensor& target,
    const mfq_tensor_backend::Tensor& source);

void restore_session_tensor(
    mfq_tensor_backend::Tensor& target,
    const mfq_tensor_backend::Tensor& source);

void restore_session_prefix_tensor(
    mfq_tensor_backend::Tensor& target,
    const mfq_tensor_backend::Tensor& source,
    int64_t dimension,
    int64_t capacity);

FullBlockSessionState capture_full_attention_session_state(
    const FullBlock& block,
    int64_t cache_pos,
    size_t& bytes);

void restore_full_attention_session_state(
    FullBlock& block,
    const FullBlockSessionState& state);

Dsv4PoolSessionState capture_dsv4_pool_session_state(
    const Dsv4PoolState& source,
    int64_t cache_pos,
    size_t& bytes);

void restore_dsv4_pool_session_state(
    Dsv4PoolState& target,
    const Dsv4PoolSessionState& state);
