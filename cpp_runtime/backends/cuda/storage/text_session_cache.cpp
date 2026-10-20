#include "storage/text_session_cache.h"
#include "core/full_block.h"
#include <array>
#include <cstring>

#include "cuda_runtime_config.h"
#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/ops.h"
#include "storage/session_state.h"
#include "core/mtp.h"
#include "mfq_paged_prefix_cache.h"
#include "paged_session_bindings.h"
#include "session_cache.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <fstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace mfq::cuda::internal {

bool cuda_has_separate_memory() {
    cudaDeviceProp properties{};
    MFQ_CUDA_CHECK(cudaGetDeviceProperties(&properties, mfq_current_cuda_device()));
    return !properties.integrated && std::strstr(properties.name, "GB10") == nullptr;
}

static std::string cuda_prefix_cache_compatibility_key(
    const mfq::ModelSource &source, int64_t max_position_embeddings) {
    std::ostringstream key;
    key << "mfq-cuda-prefix-v1\n"
        << "codec=" << (source.architecture()=="qwen4_exp" ? "cuda-qwen4-session-v1" :
            source.architecture()=="deepseek_v41" ? "cuda-deepseek-v41-session-v1" : "cuda-full-attention-kv-v1") << '\n'
        << "context=" << max_position_embeddings << '\n'
        << "architecture=" << source.architecture() << '\n';
    const auto quantization=mfq::cuda::cuda_kv_quantization_tag();
    if(!quantization.empty())key<<"kv="<<quantization<<'\n';
    for (std::size_t index = 0; index < source.source_paths().size(); ++index) {
        const auto &path = source.source_paths()[index];
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source for prefix cache: " + error.message());
        }
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error("cannot identify model source timestamp: " + error.message());
        }
        const auto modified_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(modified.time_since_epoch())
                .count();
        key << "source=" << index << ':' << path.filename().string() << ':' << size << ':'
            << modified_ns << '\n';
    }
    for (const auto &tensor : source.tensors()) {
        key << "tensor=" << tensor.name << ':' << tensor.dtype << ':' << tensor.nbytes << '\n';
    }
    return key.str();
}

std::shared_ptr<mfq::cache::PagedPrefixCache> make_cuda_paged_prefix_cache(
    const mfq::ModelSource &source, int64_t max_position_embeddings,
    bool supports_paged_text_session_state, const mfq::engine::PrefixCacheConfig &config) {
    const auto format = source.metadata().find("source.format");
    // ponytail: HF source fingerprints exclude config sidecars for now.
    if (!config.enabled ||
        (format != source.metadata().end() && format->second == "hf-safetensors") ||
        !supports_paged_text_session_state) {
        return {};
    }
    mfq::cache::PagedPrefixCacheConfig cache;
    cache.cache_dir = config.directory;
    if (!source.source_paths().empty()) cache.model_path = source.source_paths().front();
    cache.compatibility_key = cuda_prefix_cache_compatibility_key(source, max_position_embeddings);
    cache.block_size_tokens = static_cast<size_t>(config.block_tokens);
    // A longer block may lack the local/rolling state of a shorter exact
    // checkpoint. Retain those boundaries under the existing byte-budget LRU.
    cache.retain_tail_versions = 0;
    cache.max_disk_bytes = config.disk_bytes;
    const bool separate = cuda_has_separate_memory();
    cache.max_hot_bytes = separate ? config.ram_bytes : config.hot_bytes;
    cache.spill_hot_on_eviction = separate;
    if (separate) cache.available_ram_bytes = []() -> std::uint64_t {
#ifdef _WIN32
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        if (GlobalMemoryStatusEx(&state)) return state.ullAvailPhys;
#elif defined(__linux__)
        std::ifstream input("/proc/meminfo");
        std::string name, unit;
        std::uint64_t kib = 0;
        while (input >> name >> kib >> unit) if (name == "MemAvailable:") return kib * 1024;
#endif
        return UINT64_MAX;
    };
    cache.max_pending_writes = config.pending_writes;
    cache.max_pending_bytes = config.pending_bytes;
    return std::make_shared<mfq::cache::PagedPrefixCache>(std::move(cache));
}

struct CudaSessionOps {
    using Snapshot = TextSessionState;
    using Restore = TextSessionRestore;
    using Predictor = MtpModule;
    using StateError = CudaSessionStateError;
    template <class Payloads>
    static auto decode(
        const Payloads &payloads, const std::vector<int64_t> &tokens, size_t block_size) {
        if(payloads.front() && payloads.front()->size()>6 && (*payloads.front())[6]=='2')
            return decode_cuda_qwen4_paged_session(payloads,tokens,block_size);
        if(payloads.front() && payloads.front()->size()>6 && (*payloads.front())[6]=='3')
            return decode_cuda_deepseek_v41_paged_session(payloads,tokens,block_size);
        return decode_cuda_paged_session(payloads, tokens, block_size);
    }
    static auto encode(const Snapshot &state, size_t block_size, size_t index) {
        if(state.kind()==TextSessionStateKind::Qwen4)return encode_cuda_qwen4_paged_block(state,block_size,index);
        if(state.kind()==TextSessionStateKind::DeepseekV41)return encode_cuda_deepseek_v41_paged_block(state,block_size,index);
        return encode_cuda_paged_block(state, block_size, index);
    }
    static bool paged_mtp(const Snapshot& state) {
        return state.kind()==TextSessionStateKind::Qwen4 || state.kind()==TextSessionStateKind::DeepseekV41;
    }
    static bool partial_blocks(const Snapshot& state) { return paged_mtp(state); }
    static bool paged_media(const Snapshot& state) { return state.kind()==TextSessionStateKind::DeepseekV41; }
    static bool checkpoint_needs_update(const CudaPagedPayload& payload,const Snapshot& state) {
        return cuda_paged_checkpoint_needs_update(payload,state);
    }
    static size_t decodable_blocks(const std::vector<CudaPagedPayload>& payloads) { return cuda_paged_decodable_blocks(payloads); }
    static size_t token_count(const CudaPagedPayload& payload) { return cuda_paged_payload_tokens(payload); }
    static void release_host_cache() { mfq_release_host_allocator_cache(); }
};

struct TextSessionCache::Impl : mfq::engine::SessionCache<CudaSessionOps> {
    using SessionCache::SessionCache;
};

TextSessionCache::TextSessionCache(const mfq::engine::SessionCacheConfig &session_config,
    const mfq::engine::PrefixCacheConfig &prefix_config,
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache, bool supported, int disabled_reason,
    bool native_hot_snapshots)
    : impl_(std::make_unique<Impl>(
          session_config, prefix_config, std::move(paged_cache), supported, disabled_reason,
          native_hot_snapshots)) {}

TextSessionCache::~TextSessionCache() = default;

bool TextSessionCache::persistent_prefix_enabled() const noexcept {
    return impl_->persistent_prefix_enabled();
}

template <typename Model>
TextSessionRestore TextSessionCache::restore_best(Model &model, MtpModule *mtp,
    const std::string &requested_session, const std::vector<int64_t> &prompt,
    size_t maximum_prefix_tokens, const std::string &input_key) {
    return impl_->restore_best(
        model, mtp, requested_session, prompt, maximum_prefix_tokens, input_key);
}

void TextSessionCache::store(const std::string &session_id, TextSessionState state) {
    impl_->reserve_ram(state.bytes);
    impl_->store(session_id, std::move(state));
}

size_t TextSessionCache::fork_session(
    const std::string &source_session, const std::string &target_session) {
    return impl_->fork_session(source_session, target_session);
}

size_t TextSessionCache::close_session(const std::string &session_id) {
    return impl_->close_session(session_id);
}

std::vector<std::pair<std::string, double>> TextSessionCache::metrics() const {
    return impl_->metrics();
}

std::map<int, std::size_t> TextSessionCache::device_hot_bytes() const {
    std::map<std::pair<int, std::uintptr_t>, std::size_t> allocations;
    const auto tensor = [&](const mfq_tensor_backend::Tensor& value) {
        if (!value.defined() || !value.is_cuda()) return;
        auto& bytes = allocations[{value.get_device(), reinterpret_cast<std::uintptr_t>(value.data_ptr())}];
        bytes = std::max(bytes, std::size_t(value.numel()) * value.element_size());
    };
    const auto full = [&](const FullBlockSessionState& block) { tensor(block.k); tensor(block.v); };
    const auto qwen = [&](const Qwen4BlockSessionState& block) {
        tensor(block.keys); tensor(block.values); tensor(block.index); tensor(block.convolution);
        tensor(block.recurrent); tensor(block.ple_convolution);
    };
    const auto pool = [&](const Dsv4PoolSessionState& value) {
        tensor(value.state_kv); tensor(value.state_gate); tensor(value.previous_kv);
        tensor(value.previous_gate); tensor(value.pool);
    };
    impl_->visit_snapshots([&](const TextSessionState& state) {
        tensor(state.qwen4_positions);
        std::visit([&](const auto& blocks) {
            if constexpr (!std::is_same_v<std::decay_t<decltype(blocks)>, std::monostate>)
                for (const auto& block : blocks) {
                    using Block = std::decay_t<decltype(block)>;
                    if constexpr (std::is_same_v<Block, FullBlockSessionState>) full(block);
                    else if constexpr (std::is_same_v<Block, Qwen4BlockSessionState>) qwen(block);
                    else if constexpr (std::is_same_v<Block, HybridBlockSessionState>) {
                        full(block.full_attention); tensor(block.convolution_state); tensor(block.recurrent_state);
                    } else if constexpr (std::is_same_v<Block, Dsv4BlockSessionState>) {
                        tensor(block.local_cache); pool(block.compressor); pool(block.indexer_compressor);
                    } else if constexpr (std::is_same_v<Block, DeepseekV41BlockSessionState>) {
                        tensor(block.local_kv); tensor(block.compressed_kv); tensor(block.index_k);
                        tensor(block.partial_kv); tensor(block.partial_score);
                    } else { tensor(block.kv_cache); tensor(block.index_cache); }
                }
        }, state.payload);
        if (state.mtp) {
            tensor(state.mtp->last_target_hidden);
            for (const auto& block : state.mtp->blocks) full(block);
            for (const auto& block : state.mtp->qwen4_blocks) qwen(block);
            for (const auto& ring : state.mtp->deepseek_v41_rings) tensor(ring);
            for (const auto& value : state.mtp->qwen4_positions) tensor(value);
        }
    });
    std::map<int, std::size_t> result;
    for (const auto& [address, bytes] : allocations) result[address.first] += bytes;
    return result;
}

size_t TextSessionCache::clear_live_sessions() noexcept { return impl_->clear_live_sessions(); }

size_t TextSessionCache::clear() { return impl_->clear(); }

uint64_t TextSessionCache::trim_hot(uint64_t target_bytes) { return impl_->trim_hot(target_bytes); }
uint64_t TextSessionCache::set_hot_limit(uint64_t max_bytes) { return impl_->set_hot_limit(max_bytes); }
uint64_t TextSessionCache::set_ram_limit(uint64_t max_bytes) { return impl_->set_ram_limit(max_bytes); }
void TextSessionCache::reserve_ram(uint64_t bytes) { impl_->reserve_ram(bytes); }
uint64_t TextSessionCache::set_disk_limit(uint64_t max_bytes) { return impl_->set_disk_limit(max_bytes); }
uint64_t TextSessionCache::refresh_disk_index() { return impl_->refresh_disk_index(); }

#define MFQ_INSTANTIATE_SESSION_CACHE(MODEL)                                                       \
    template TextSessionRestore TextSessionCache::restore_best(MODEL &,                            \
        MtpModule *,                                                                               \
        const std::string &,                                                                       \
        const std::vector<int64_t> &,                                                              \
        size_t,                                                                                    \
        const std::string &);

MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Qwen35CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::MiniCPMO45CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::MiniCPMOTtsCausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Gemma4CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::GlmDsaCausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Glm5CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Qwen4CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::DeepseekV4CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::DeepseekV41CausalLm)

#undef MFQ_INSTANTIATE_SESSION_CACHE

} // namespace mfq::cuda::internal

void CudaPagedWriter::raw(const void *data, size_t size) {
    const auto *begin = static_cast<const uint8_t *>(data);
    bytes_.insert(bytes_.end(), begin, begin + size);
}

void CudaPagedWriter::tensor(const mfq_tensor_backend::Tensor &source) {
    if (!source.defined() || !source.is_cuda() || source.dim() <= 0 || source.dim() > 8) {
        throw std::runtime_error("invalid CUDA paged cache tensor");
    }
    const auto dtype = source.scalar_type();
    if (dtype != mfq_tensor_backend::kFloat16 && dtype != mfq_tensor_backend::kBFloat16 &&
        dtype != mfq_tensor_backend::kFloat32 && dtype != mfq_tensor_backend::kInt32 && dtype != mfq_tensor_backend::kInt64) {
        throw std::runtime_error("unsupported CUDA paged cache tensor dtype");
    }
    const auto cpu = source.contiguous().to(mfq_tensor_backend::kCPU);
    scalar<int32_t>(static_cast<int32_t>(dtype));
    scalar<int32_t>(source.get_device());
    scalar<uint32_t>(static_cast<uint32_t>(cpu.dim()));
    scalar<uint32_t>(0);
    for (const auto dimension : cpu.sizes())
        scalar<int64_t>(dimension);
    const auto bytes = static_cast<uint64_t>(cpu.numel() * cpu.element_size());
    scalar<uint64_t>(bytes);
    raw(cpu.data_ptr(), static_cast<size_t>(bytes));
}

CudaPagedPayload CudaPagedWriter::finish() && {
    return std::make_shared<const std::vector<uint8_t>>(std::move(bytes_));
}

void CudaPagedReader::raw(void *destination, size_t size, const char *name) {
    if (remaining() < size) {
        throw CudaSessionStateError(std::string("truncated CUDA paged cache ") + name);
    }
    std::memcpy(destination, cursor_, size);
    cursor_ += size;
}

std::pair<mfq_tensor_backend::Tensor, int> CudaPagedReader::tensor() {
    const auto dtype_value = scalar<int32_t>("dtype");
    const auto device = scalar<int32_t>("device");
    const auto rank = scalar<uint32_t>("rank");
    (void)scalar<uint32_t>("tensor flags");
    if (rank == 0 || rank > 8 || device < 0) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor header");
    }
    const auto dtype = static_cast<mfq_tensor_backend::ScalarType>(dtype_value);
    if (dtype != mfq_tensor_backend::kFloat16 && dtype != mfq_tensor_backend::kBFloat16 &&
        dtype != mfq_tensor_backend::kFloat32 && dtype != mfq_tensor_backend::kInt32 && dtype != mfq_tensor_backend::kInt64) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor dtype");
    }
    std::vector<int64_t> shape;
    shape.reserve(rank);
    uint64_t elements = 1;
    for (uint32_t index = 0; index < rank; ++index) {
        const auto dimension = scalar<int64_t>("shape");
        if (dimension <= 0 ||
            elements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dimension)) {
            throw CudaSessionStateError("invalid CUDA paged cache tensor shape");
        }
        shape.push_back(dimension);
        elements *= static_cast<uint64_t>(dimension);
    }
    const auto bytes = scalar<uint64_t>("tensor size");
    const uint64_t element_size = dtype == mfq_tensor_backend::kInt64 ? 8 :
        (dtype == mfq_tensor_backend::kFloat32 || dtype == mfq_tensor_backend::kInt32) ? 4 : 2;
    if (elements > std::numeric_limits<uint64_t>::max() / element_size ||
        bytes != elements * element_size || bytes > remaining()) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor size");
    }
    auto cpu = mfq_tensor_backend::empty(
        shape, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(dtype));
    raw(cpu.data_ptr(), static_cast<size_t>(bytes), "tensor");
    return {std::move(cpu), device};
}

void CudaPagedReader::expect_end() const {
    if (cursor_ != end_) {
        throw CudaSessionStateError("trailing CUDA paged cache payload bytes");
    }
}

namespace {
constexpr std::array<uint8_t,8> qwen4_magic{'M','F','Q','C','U','D','2',0};
struct Qwen4PagedHeader { uint32_t start,count,layers;bool exact; };
Qwen4PagedHeader read_qwen4_header(CudaPagedReader& reader) {
    std::array<uint8_t,8> magic{};reader.raw(magic.data(),magic.size(),"magic");
    if(magic!=qwen4_magic || reader.scalar<uint32_t>("version")!=1)
        throw CudaSessionStateError("unsupported Qwen4 prefix cache payload");
    Qwen4PagedHeader header{reader.scalar<uint32_t>("start"),reader.scalar<uint32_t>("count"),
        reader.scalar<uint32_t>("layers"),reader.scalar<uint32_t>("exact boundary")!=0};
    if(!header.count || !header.layers || header.layers>4096)throw CudaSessionStateError("invalid Qwen4 prefix geometry");
    return header;
}
void write_optional(CudaPagedWriter& writer,const mfq_tensor_backend::Tensor& tensor) {
    writer.scalar<uint32_t>(tensor.defined()?1:0);if(tensor.defined())writer.tensor(tensor);
}
mfq_tensor_backend::Tensor read_optional(CudaPagedReader& reader) {
    if(!reader.scalar<uint32_t>("tensor presence"))return {};
    auto [tensor,device]=reader.tensor();
    return tensor.to(mfq_tensor_backend::Device{mfq_tensor_backend::DeviceType::cuda,device});
}
void write_recurrent(CudaPagedWriter& writer,const Qwen4BlockSessionState& layer) {
    write_optional(writer,layer.convolution);write_optional(writer,layer.recurrent);write_optional(writer,layer.ple_convolution);
    writer.scalar<int64_t>(layer.ple_batch);writer.scalar<uint32_t>(uint32_t(layer.ple_tokens.size()));
    for(auto token:layer.ple_tokens)writer.scalar<int64_t>(token);
}
void read_recurrent(CudaPagedReader& reader,Qwen4BlockSessionState& layer) {
    layer.convolution=read_optional(reader);layer.recurrent=read_optional(reader);layer.ple_convolution=read_optional(reader);
    layer.ple_batch=reader.scalar<int64_t>("PLE batch");
    const auto count=reader.scalar<uint32_t>("PLE history length");
    if(count>65536)throw CudaSessionStateError("invalid Qwen4 PLE history");
    for(uint32_t i=0;i<count;++i)layer.ple_tokens.push_back(reader.scalar<int64_t>("PLE token"));
}
void write_qwen4_mtp(CudaPagedWriter& writer,const MtpSessionState& state) {
    writer.scalar<int64_t>(state.cache_pos);write_optional(writer,state.last_target_hidden);
    writer.scalar<uint32_t>(uint32_t(state.qwen4_blocks.size()));
    for(std::size_t i=0;i<state.qwen4_blocks.size();++i) {
        const auto& layer=state.qwen4_blocks[i];
        write_optional(writer,layer.keys);write_optional(writer,layer.values);write_optional(writer,layer.index);
        write_recurrent(writer,layer);write_optional(writer,state.qwen4_positions[i]);
        writer.scalar<int64_t>(state.qwen4_lengths[i]);
    }
}
std::size_t qwen4_layer_bytes(const Qwen4BlockSessionState& layer) {
    std::size_t bytes=layer.ple_tokens.size()*sizeof(int64_t);
    for(const auto* tensor:{&layer.keys,&layer.values,&layer.index,&layer.convolution,&layer.recurrent,&layer.ple_convolution})
        bytes+=session_tensor_bytes(*tensor);
    return bytes;
}
MtpSessionState read_qwen4_mtp(CudaPagedReader& reader) {
    MtpSessionState state;state.cache_pos=reader.scalar<int64_t>("MTP position");state.last_target_hidden=read_optional(reader);
    const auto count=reader.scalar<uint32_t>("MTP layers");
    if(!count || count>64)throw CudaSessionStateError("invalid Qwen4 MTP layers");
    state.bytes=session_tensor_bytes(state.last_target_hidden);
    for(uint32_t i=0;i<count;++i) {
        Qwen4BlockSessionState layer;layer.keys=read_optional(reader);layer.values=read_optional(reader);layer.index=read_optional(reader);
        read_recurrent(reader,layer);state.bytes+=qwen4_layer_bytes(layer);state.qwen4_blocks.push_back(std::move(layer));
        auto positions=read_optional(reader);state.bytes+=session_tensor_bytes(positions);state.qwen4_positions.push_back(std::move(positions));
        state.qwen4_lengths.push_back(reader.scalar<int64_t>("MTP layer position"));
    }
    return state;
}
}

CudaPagedPayload encode_cuda_qwen4_paged_block(const TextSessionState& state,size_t block_size,size_t index) {
    const auto* layers=std::get_if<std::vector<Qwen4BlockSessionState>>(&state.payload);
    if(!block_size || !layers || layers->empty() || state.cache_pos<=0 || state.tokens.size()!=std::size_t(state.cache_pos) ||
        index>state.tokens.size()/block_size || index*block_size>=state.tokens.size() || state.cache_pos>UINT32_MAX)
        throw std::runtime_error("invalid Qwen4 prefix block boundary");
    const auto start=index*block_size,count=std::min(block_size,state.tokens.size()-start);
    const bool exact=start+count==state.tokens.size();
    CudaPagedWriter writer;writer.raw(qwen4_magic.data(),qwen4_magic.size());writer.scalar<uint32_t>(1);
    writer.scalar<uint32_t>(uint32_t(start));writer.scalar<uint32_t>(uint32_t(count));
    writer.scalar<uint32_t>(uint32_t(layers->size()));writer.scalar<uint32_t>(exact?1:0);
    for(const auto& layer:*layers) {
        for(const auto* tensor:{&layer.keys,&layer.values,&layer.index})
            write_optional(writer,tensor->defined()?tensor->narrow(1,int64_t(start),int64_t(count)):mfq_tensor_backend::Tensor{});
        if(exact)write_recurrent(writer,layer);
    }
    if(exact) {
        write_optional(writer,state.qwen4_positions);writer.scalar<uint32_t>(state.mtp?1:0);
        if(state.mtp)write_qwen4_mtp(writer,*state.mtp);
    }
    return std::move(writer).finish();
}

size_t cuda_paged_payload_tokens(const CudaPagedPayload& payload) {
    if(!payload)throw CudaSessionStateError("null CUDA prefix block");
    CudaPagedReader reader(*payload);std::array<uint8_t,8> magic{};reader.raw(magic.data(),magic.size(),"magic");
    constexpr std::array<uint8_t,8> full_magic{'M','F','Q','C','U','D','1',0};
    constexpr std::array<uint8_t,8> deepseek_magic{'M','F','Q','C','U','D','3',0};
    if((magic!=qwen4_magic && magic!=full_magic && magic!=deepseek_magic) || reader.scalar<uint32_t>("version")!=1)
        throw CudaSessionStateError("invalid CUDA prefix block header");
    (void)reader.scalar<uint32_t>("start");return reader.scalar<uint32_t>("count");
}
size_t cuda_paged_decodable_blocks(const std::vector<CudaPagedPayload>& payloads) {
    for(size_t count=payloads.size();count>0;--count) {
        const auto& payload=payloads[count-1];
        (void)cuda_paged_payload_tokens(payload);
        if((*payload)[6]=='1')return count;
        if((*payload)[6]=='3') {
            if(cuda_deepseek_v41_paged_exact(payload))return count;
            continue;
        }
        CudaPagedReader reader(*payload);if(read_qwen4_header(reader).exact)return count;
    }
    return 0;
}

TextSessionState decode_cuda_qwen4_paged_session(const std::vector<CudaPagedPayload>& payloads,
    const std::vector<int64_t>& tokens,size_t block_size) {
    namespace tb=mfq_tensor_backend;
    TextSessionState state;state.tokens=tokens;state.cache_pos=int64_t(tokens.size());
    std::vector<Qwen4BlockSessionState> layers;
    std::vector<std::array<std::vector<tb::Tensor>,3>> fragments;
    size_t position=0;
    for(size_t i=0;i<payloads.size();++i) {
        if(!payloads[i])throw CudaSessionStateError("null Qwen4 prefix payload");
        CudaPagedReader reader(*payloads[i]);const auto header=read_qwen4_header(reader);
        if(header.start!=position || header.count>block_size || (i+1<payloads.size() && header.count!=block_size) ||
            (i && header.layers!=layers.size()))throw CudaSessionStateError("incompatible Qwen4 prefix chain");
        if(!i){layers.resize(header.layers);fragments.resize(header.layers);}
        position+=header.count;
        for(size_t layer=0;layer<layers.size();++layer) {
            for(size_t field=0;field<3;++field) {
                auto tensor=read_optional(reader);
                if(tensor.defined()) {
                    if(tensor.dim()!=3 || tensor.size(0)!=1 || tensor.size(1)!=header.count)
                        throw CudaSessionStateError("invalid Qwen4 QSA prefix geometry");
                    fragments[layer][field].push_back(std::move(tensor));
                }
            }
            if(header.exact) {
                layers[layer].convolution={};layers[layer].recurrent={};layers[layer].ple_convolution={};layers[layer].ple_tokens.clear();
                read_recurrent(reader,layers[layer]);
            }
        }
        if(header.exact) {
            state.qwen4_positions=read_optional(reader);
            state.mtp.reset();if(reader.scalar<uint32_t>("MTP presence"))state.mtp=read_qwen4_mtp(reader);
        }
        if(i+1==payloads.size() && !header.exact)throw CudaSessionStateError("Qwen4 prefix has no exact recurrent checkpoint");
        reader.expect_end();
    }
    if(payloads.empty() || position!=tokens.size() || !state.qwen4_positions.defined() || state.qwen4_positions.size(-1)!=int64_t(position))
        throw CudaSessionStateError("Qwen4 prefix boundary mismatch");
    for(size_t i=0;i<layers.size();++i) {
        auto& layer=layers[i];std::array<tb::Tensor*,3> fields{&layer.keys,&layer.values,&layer.index};
        for(size_t j=0;j<3;++j)if(!fragments[i][j].empty()) {
            if(fragments[i][j].size()!=payloads.size())throw CudaSessionStateError("incomplete Qwen4 QSA prefix chain");
            *fields[j]=fragments[i][j].size()==1?fragments[i][j].front():tb::cat(fragments[i][j],1);
        }
        state.bytes+=qwen4_layer_bytes(layer);
    }
    state.bytes+=session_tensor_bytes(state.qwen4_positions)+(state.mtp?state.mtp->bytes:0);
    state.payload=std::move(layers);return state;
}

size_t CudaPagedReader::remaining() const { return static_cast<size_t>(end_ - cursor_); }



std::vector<CudaPagedPayload> encode_cuda_paged_session(const TextSessionState &state,
                                                        size_t block_size, size_t first_block,
                                                        size_t maximum_blocks) {
    static constexpr std::array<uint8_t, 8> magic{'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    const auto *layers = std::get_if<std::vector<FullBlockSessionState>>(&state.payload);
    // Paged prefixes are text-only; media positions are not block-sliceable.
    if (state.kind() != TextSessionStateKind::FullAttention || state.decode_position_delta != 0 ||
        !state.input_key.empty() || state.cache_pos <= 0 || layers == nullptr || layers->empty() ||
        state.tokens.size() != static_cast<size_t>(state.cache_pos)) {
        throw std::runtime_error("CUDA session state is not block-sliceable");
    }
    const size_t full_blocks = state.tokens.size() / block_size;
    if (first_block > full_blocks) {
        throw std::runtime_error("invalid CUDA cache first block");
    }
    const size_t end_block =
        maximum_blocks > full_blocks - first_block ? full_blocks : first_block + maximum_blocks;
    std::vector<CudaPagedPayload> result;
    result.reserve(end_block - first_block);
    for (size_t block_index = first_block; block_index < end_block; ++block_index) {
        const int64_t start = static_cast<int64_t>(block_index * block_size);
        CudaPagedWriter writer;
        writer.raw(magic.data(), magic.size());
        writer.scalar<uint32_t>(1);
        writer.scalar<uint32_t>(static_cast<uint32_t>(start));
        writer.scalar<uint32_t>(static_cast<uint32_t>(block_size));
        writer.scalar<uint32_t>(static_cast<uint32_t>(layers->size()));
        for (const auto &layer : *layers) {
            if (layer.ring || layer.capacity <= 0 || !layer.k.defined() || !layer.v.defined() ||
                !session_kv_geometry_matches(layer.k,layer.v) ||
                layer.k.size(2) < start + static_cast<int64_t>(block_size)) {
                throw std::runtime_error("CUDA session layer cannot be block-sliced");
            }
            writer.scalar<int64_t>(layer.capacity);
            writer.tensor(layer.k.narrow(2, start, static_cast<int64_t>(block_size)));
            writer.tensor(layer.v.narrow(2, start, static_cast<int64_t>(block_size)));
        }
        result.push_back(std::move(writer).finish());
    }
    return result;
}

CudaPagedPayload encode_cuda_paged_block(const TextSessionState &state, size_t block_size,
                                         size_t block_index) {
    auto result = encode_cuda_paged_session(state, block_size, block_index, 1);
    if (result.size() != 1) {
        throw std::runtime_error("invalid CUDA cache block index");
    }
    return std::move(result.front());
}

CudaDecodedPagedBlock decode_cuda_paged_block(const std::vector<uint8_t> &payload) {
    static constexpr std::array<uint8_t, 8> magic{'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    CudaPagedReader reader(payload);
    std::array<uint8_t, 8> found_magic{};
    reader.raw(found_magic.data(), found_magic.size(), "magic");
    if (found_magic != magic || reader.scalar<uint32_t>("version") != 1) {
        throw CudaSessionStateError("unsupported CUDA paged cache payload");
    }
    CudaDecodedPagedBlock block;
    block.start = reader.scalar<uint32_t>("start");
    block.count = reader.scalar<uint32_t>("count");
    const auto layer_count = reader.scalar<uint32_t>("layer count");
    if (block.count == 0 || layer_count == 0 || layer_count > 4096) {
        throw CudaSessionStateError("invalid CUDA paged cache geometry");
    }
    block.layers.reserve(layer_count);
    for (uint32_t index = 0; index < layer_count; ++index) {
        CudaDecodedPagedLayer layer;
        layer.capacity = reader.scalar<int64_t>("capacity");
        auto key = reader.tensor();
        auto value = reader.tensor();
        if (key.second != value.second || !session_kv_geometry_matches(key.first,value.first) ||
            key.first.size(2) != static_cast<int64_t>(block.count)) {
            throw CudaSessionStateError("invalid CUDA paged cache layer");
        }
        layer.device = key.second;
        layer.k = std::move(key.first);
        layer.v = std::move(value.first);
        block.layers.push_back(std::move(layer));
    }
    reader.expect_end();
    return block;
}

TextSessionState decode_cuda_paged_session(const std::vector<CudaPagedPayload> &payloads,
                                           const std::vector<int64_t> &tokens, size_t block_size) {
    if (payloads.empty() || tokens.size() != payloads.size() * block_size) {
        throw CudaSessionStateError("CUDA paged cache chain length mismatch");
    }
    std::vector<CudaDecodedPagedBlock> blocks;
    blocks.reserve(payloads.size());
    size_t expected_start = 0;
    size_t layer_count = 0;
    for (const auto &payload : payloads) {
        if (!payload) {
            throw CudaSessionStateError("CUDA paged cache block payload is null");
        }
        auto block = decode_cuda_paged_block(*payload);
        if (block.start != expected_start || block.count != block_size ||
            (layer_count != 0 && block.layers.size() != layer_count)) {
            throw CudaSessionStateError("incompatible CUDA paged cache chain");
        }
        expected_start += block.count;
        layer_count = block.layers.size();
        blocks.push_back(std::move(block));
    }
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = static_cast<int64_t>(tokens.size());
    state.payload = std::vector<FullBlockSessionState>{};
    auto &layers = std::get<std::vector<FullBlockSessionState>>(state.payload);
    layers.reserve(layer_count);
    for (size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        const auto &final = blocks.back().layers[layer_index];
        std::vector<mfq_tensor_backend::Tensor> keys;
        std::vector<mfq_tensor_backend::Tensor> values;
        keys.reserve(blocks.size());
        values.reserve(blocks.size());
        for (const auto &block : blocks) {
            const auto &layer = block.layers[layer_index];
            if (layer.capacity != final.capacity || layer.device != final.device ||
                layer.k.scalar_type() != final.k.scalar_type() ||
                layer.k.size(0) != final.k.size(0) || layer.k.size(1) != final.k.size(1) ||
                layer.k.size(3) != final.k.size(3) || layer.v.size(3)!=final.v.size(3)) {
                throw CudaSessionStateError("inconsistent CUDA paged cache topology");
            }
            keys.push_back(layer.k);
            values.push_back(layer.v);
        }
        auto key = mfq_tensor_backend::cat(keys, 2).contiguous();
        auto value = mfq_tensor_backend::cat(values, 2).contiguous();
        MfqCudaGuard guard(final.device);
        const auto options = key.options().device(
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, final.device));
        key = key.to(options, true, false).contiguous();
        value = value.to(options, true, false).contiguous();
        state.bytes += static_cast<size_t>(key.numel() * key.element_size() +
                                           value.numel() * value.element_size());
        layers.push_back(FullBlockSessionState{
            std::move(key),
            std::move(value),
            final.capacity,
            false,
        });
    }
    return state;
}

size_t session_tensor_bytes(const mfq_tensor_backend::Tensor &tensor) {
    if (!tensor.defined())
        return 0;
    return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

bool session_tensor_layout_matches(const mfq_tensor_backend::Tensor &target,
                                   const mfq_tensor_backend::Tensor &source) {
    return target.defined() && source.defined() && target.sizes() == source.sizes() &&
           target.scalar_type() == source.scalar_type() && target.device() == source.device();
}

void restore_session_tensor(mfq_tensor_backend::Tensor &target,
                            const mfq_tensor_backend::Tensor &source) {
    if (!source.defined()) {
        target = mfq_tensor_backend::Tensor();
        return;
    }
    if (!session_tensor_layout_matches(target, source)) {
        target = mfq_tensor_backend::empty(source.sizes(), source.options());
    }
    target.copy_(source);
}

void restore_session_prefix_tensor(mfq_tensor_backend::Tensor &target,
                                   const mfq_tensor_backend::Tensor &source, int64_t dimension,
                                   int64_t capacity) {
    if (!source.defined() || dimension < 0 || dimension >= source.dim() || capacity <= 0 ||
        source.size(dimension) > capacity) {
        throw CudaSessionStateError("session prefix tensor layout is invalid");
    }
    auto target_shape = source.sizes().vec();
    target_shape.at(static_cast<size_t>(dimension)) = capacity;
    const bool compatible =
        target.defined() && target.sizes() == mfq_tensor_backend::IntArrayRef(target_shape) &&
        target.scalar_type() == source.scalar_type() && target.device() == source.device();
    if (!compatible) {
        target = mfq_tensor_backend::empty(target_shape, source.options());
    }
    if (source.size(dimension) > 0) {
        target.narrow(dimension, 0, source.size(dimension)).copy_(source);
    }
}

FullBlockSessionState capture_full_attention_session_state(const FullBlock &block,
                                                           int64_t cache_pos, size_t &bytes) {
    if (!block.cache.k.defined() || !block.cache.v.defined() || block.cache.k.dim() != 4 ||
        !session_kv_geometry_matches(block.cache.k,block.cache.v) || block.cache.k.size(0) != 1) {
        throw std::runtime_error("full-attention KV cache is unavailable");
    }
    FullBlockSessionState state;
    state.capacity = block.cache.k.size(2);
    state.ring = block.cache.ring;
    const int64_t saved_tokens =
        state.ring ? state.capacity : std::min<int64_t>(cache_pos, state.capacity);
    state.k = block.cache.k.narrow(2, 0, saved_tokens).clone();
    state.v = block.cache.v.narrow(2, 0, saved_tokens).clone();
    bytes += session_tensor_bytes(state.k);
    bytes += session_tensor_bytes(state.v);
    return state;
}

void restore_full_attention_session_state(FullBlock &block, const FullBlockSessionState &state) {
    if (!state.k.defined() || !state.v.defined() || state.capacity <= 0 || state.k.dim() != 4 ||
        !session_kv_geometry_matches(state.k,state.v) || state.k.size(0) != 1 ||
        state.k.size(2) > state.capacity) {
        throw CudaSessionStateError("full-attention session KV layout is invalid");
    }
    restore_session_prefix_tensor(block.cache.k, state.k, 2, state.capacity);
    restore_session_prefix_tensor(block.cache.v, state.v, 2, state.capacity);
    block.cache.ring = state.ring;
    block.cache.quantization = block.kv_quantization;
}
