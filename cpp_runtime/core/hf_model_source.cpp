#include "mfq/hf_model_source.h"

#include "mfq/fp8_sq_blob.h"
#include "mfq_legacy_tensor_names.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace mfq {
namespace {

using json = nlohmann::json;

constexpr std::uint64_t kMxHeaderBytes = 56;
constexpr std::uint32_t kMaxRecords = std::uint32_t{1} << 20;

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    std::string_view what) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string(what) + " byte size overflow");
    }
    return left + right;
}

template <typename T>
void append_little(std::vector<std::byte>& destination, T value) {
    using Unsigned = std::make_unsigned_t<T>;
    const auto bits = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        destination.push_back(static_cast<std::byte>(bits >> (index * 8)));
    }
}

std::uint64_t checked_product(
    const std::vector<std::int64_t>& shape,
    std::uint64_t item_size,
    const std::string& name) {
    auto result = item_size;
    for (const auto dimension : shape) {
        if (dimension <= 0 ||
            static_cast<std::uint64_t>(dimension) >
                std::numeric_limits<std::uint64_t>::max() / result) {
            throw std::runtime_error("invalid Safetensors tensor shape: " + name);
        }
        result *= static_cast<std::uint64_t>(dimension);
    }
    return result;
}

std::uint64_t dense_item_size(std::string_view dtype) {
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "F32" || dtype == "I32") return 4;
    if (dtype == "I64") return 8;
    return 0;
}

std::vector<std::byte> dense_prefix(
    const std::vector<std::int64_t>& shape) {
    std::vector<std::byte> result;
    result.reserve(4 + shape.size() * 8);
    append_little<std::uint32_t>(
        result, static_cast<std::uint32_t>(shape.size()));
    for (const auto dimension : shape) {
        append_little<std::int64_t>(result, dimension);
    }
    return result;
}

std::vector<std::byte> fp8_128_sq8_prefix(
    const std::vector<std::int64_t>& shape,
    std::string_view scale_dtype,
    const std::vector<std::int64_t>& scale_shape,
    const std::string& name) {
    if (shape.size() != 2 || scale_shape.size() != 2 ||
        shape[0] <= 0 || shape[1] <= 0 ||
        scale_shape != std::vector<std::int64_t>{
            (shape[0] + 127) / 128, (shape[1] + 127) / 128}) {
        throw std::runtime_error(
            "invalid native FP8-128 scale geometry: " + name);
    }
    mfq::fp8sq::ScaleKind scale_kind;
    if (scale_dtype == "BF16") {
        scale_kind = mfq::fp8sq::ScaleKind::Bf16;
    } else if (scale_dtype == "F16") {
        scale_kind = mfq::fp8sq::ScaleKind::F16;
    } else if (scale_dtype == "F32") {
        scale_kind = mfq::fp8sq::ScaleKind::F32;
    } else {
        throw std::runtime_error(
            "invalid native FP8-128 scale dtype: " + name);
    }

    std::vector<std::byte> result;
    const auto selector_bytes =
        ((static_cast<std::size_t>(shape[0]) * 3 + 7) / 8 + 3) &
        ~std::size_t{3};
    result.reserve(
        mfq::fp8sq::kHeaderBytes + selector_bytes +
        mfq::fp8sq::kPaletteBytes);
    for (const char value : std::string_view("F8SQ")) {
        result.push_back(static_cast<std::byte>(value));
    }
    result.push_back(std::byte{1});
    result.push_back(static_cast<std::byte>(scale_kind));
    append_little<std::uint16_t>(result, 0);
    append_little<std::uint16_t>(result, 128);
    append_little<std::uint16_t>(result, 128);
    append_little<std::uint64_t>(
        result, static_cast<std::uint64_t>(shape[0]));
    append_little<std::uint64_t>(
        result, static_cast<std::uint64_t>(shape[1]));
    append_little<std::uint64_t>(
        result, static_cast<std::uint64_t>(scale_shape[0]));
    append_little<std::uint64_t>(
        result, static_cast<std::uint64_t>(scale_shape[1]));

    const auto selector_offset = result.size();
    result.resize(selector_offset + selector_bytes, std::byte{0});
    for (std::size_t row = 0; row < static_cast<std::size_t>(shape[0]); ++row) {
        const auto bit = row * 3;
        const auto byte = selector_offset + bit / 8;
        const auto shift = static_cast<unsigned>(bit & 7);
        result[byte] |= static_cast<std::byte>(7u << shift);
        if (shift + 3 > 8) {
            result[byte + 1] |= static_cast<std::byte>(7u >> (8 - shift));
        }
    }

    const auto palette_offset = result.size();
    result.resize(palette_offset + mfq::fp8sq::kPaletteBytes, std::byte{0});
    std::size_t destination = palette_offset;
    for (unsigned bits = 1; bits <= 7; ++bits) {
        const auto count = std::size_t{1} << bits;
        for (unsigned code = 0, written = 0; written < count; ++code) {
            if ((code & 0x7Fu) == 0x7Fu) continue;
            result[destination++] = static_cast<std::byte>(code);
            ++written;
        }
    }
    return result;
}

std::vector<std::byte> mx_prefix(
    std::string_view dtype,
    const std::vector<std::int64_t>& logical_shape,
    const std::vector<std::int64_t>& storage_shape,
    const std::vector<std::int64_t>& scale_shape,
    const std::string& name) {
    if (logical_shape.size() != 2 || storage_shape.size() != 2 ||
        scale_shape.size() != 2) {
        throw std::runtime_error("native MX tensor must be rank two: " + name);
    }
    const auto rows = logical_shape[0];
    const auto columns = logical_shape[1];
    if (rows <= 0 || columns <= 0 || columns % 32 != 0) {
        throw std::runtime_error("invalid native MX tensor shape: " + name);
    }
    const std::vector<std::int64_t> expected_storage = dtype == "MXFP4"
        ? std::vector<std::int64_t>{rows, columns / 2}
        : std::vector<std::int64_t>{rows, columns};
    bool valid_scales = false;
    if (dtype == "MXFP4") {
        valid_scales = scale_shape ==
            std::vector<std::int64_t>{rows, columns / 32};
    } else {
        valid_scales =
            scale_shape == std::vector<std::int64_t>{rows, columns / 32} ||
            scale_shape == std::vector<std::int64_t>{
                (rows + 31) / 32, columns / 32} ||
            (columns % 128 == 0 &&
             scale_shape == std::vector<std::int64_t>{
                 (rows + 127) / 128, columns / 128});
    }
    if (storage_shape != expected_storage || !valid_scales) {
        throw std::runtime_error("invalid native MX storage geometry: " + name);
    }
    std::vector<std::byte> result;
    result.reserve(kMxHeaderBytes);
    for (const char value : std::string_view("MXT1")) {
        result.push_back(static_cast<std::byte>(value));
    }
    result.push_back(std::byte{1});
    result.push_back(dtype == "MXFP4" ? std::byte{4} : std::byte{8});
    append_little<std::uint16_t>(result, 0);
    for (const auto dimension : logical_shape) {
        append_little<std::uint64_t>(
            result, static_cast<std::uint64_t>(dimension));
    }
    for (const auto dimension : storage_shape) {
        append_little<std::uint64_t>(
            result, static_cast<std::uint64_t>(dimension));
    }
    for (const auto dimension : scale_shape) {
        append_little<std::uint64_t>(
            result, static_cast<std::uint64_t>(dimension));
    }
    return result;
}

std::unordered_map<std::string, std::string> source_map(
    const HfSafetensorsSource& source) {
    if (source.has_asset(kHfSourceMapAsset)) {
        const auto bytes = source.read_asset(kHfSourceMapAsset);
        json payload;
        try {
            payload = json::parse(
                reinterpret_cast<const char*>(bytes.data()),
                reinterpret_cast<const char*>(bytes.data() + bytes.size()));
        } catch (const json::exception& error) {
            throw std::runtime_error(
                std::string("invalid native HF source map: ") + error.what());
        }
        const auto aliases = payload.find("canonical_to_source");
        if (!payload.is_object() ||
            payload.value("schema", std::string{}) != "mfq.hf-source-map" ||
            payload.value("version", 0) != 1 || aliases == payload.end() ||
            !aliases->is_object()) {
            throw std::runtime_error("unsupported native HF source map contract");
        }
        std::unordered_map<std::string, std::string> result;
        for (const auto& [canonical, raw_source] : aliases->items()) {
            if (canonical.empty() || !raw_source.is_string() ||
                raw_source.get_ref<const std::string&>().empty() ||
                !result.emplace(canonical, raw_source.get<std::string>()).second) {
                throw std::runtime_error(
                    "native HF source map has an invalid entry");
            }
        }
        return result;
    }

    if (!source.has_asset(kModelConfigAsset)) return {};
    const auto config = source.read_asset(kModelConfigAsset);
    std::vector<std::string> names;
    names.reserve(source.tensors().size());
    for (const auto& tensor : source.tensors()) names.push_back(tensor.name);
    auto compatibility = make_legacy_tensor_aliases(
        source.architecture(),
        std::string_view(reinterpret_cast<const char*>(config.data()), config.size()),
        names).canonical_to_stored;
    std::unordered_set<std::string> claimed;
    for (const auto& [canonical, stored] : compatibility) {
        static_cast<void>(canonical);
        claimed.insert(stored);
    }
    for (const auto& name : names) {
        if (claimed.find(name) == claimed.end()) {
            compatibility.emplace(name, name);
        }
    }
    return compatibility;
}

} // namespace

struct HfModelSource::Impl {
    struct Segment {
        std::uint64_t offset = 0;
        std::vector<std::byte> inline_bytes;
        std::string source_name;
        std::uint64_t source_offset = 0;
        std::uint64_t nbytes = 0;
    };
    struct Record {
        TensorMetadata metadata;
        std::vector<Segment> segments;
    };

    explicit Impl(std::filesystem::path root)
        : source(std::move(root)) {}

    HfSafetensorsSource source;
    std::vector<TensorMetadata> tensors;
    std::unordered_map<std::string, std::size_t> tensor_indices;
    std::unordered_map<std::string, Record> records;
};

HfModelSource::HfModelSource(std::filesystem::path root)
    : impl_(std::make_unique<Impl>(std::move(root))) {
    const bool has_explicit_source_map =
        impl_->source.has_asset(kHfSourceMapAsset);
    auto canonical_to_source = source_map(impl_->source);
    if (canonical_to_source.empty()) {
        for (const auto& tensor : impl_->source.tensors()) {
            canonical_to_source.emplace(tensor.name, tensor.name);
        }
    }
    std::unordered_map<std::string, std::string> source_to_canonical;
    for (const auto& [canonical, source] : canonical_to_source) {
        if (canonical.empty() || source.empty() ||
            impl_->source.find_tensor(source) == nullptr) {
            throw std::runtime_error("invalid HF canonical tensor mapping: " +
                                     canonical);
        }
        const auto [found, inserted] =
            source_to_canonical.emplace(source, canonical);
        if (!inserted && found->second != canonical) {
            throw std::runtime_error(
                "native HF source tensor has multiple canonical owners: " + source);
        }
    }

    const auto scale_candidates = [&](
            const std::string& source_name,
            const std::string& canonical_name) {
        std::vector<std::string> candidates;
        if (ends_with(canonical_name, ".weight")) {
            const auto base = canonical_name.substr(0, canonical_name.size() - 7);
            for (const auto suffix : {".weight_scale_inv", ".weight_scale",
                                      ".weight_scale_2", ".scale"}) {
                const auto found = canonical_to_source.find(base + suffix);
                if (found != canonical_to_source.end()) {
                    candidates.push_back(found->second);
                }
            }
        }
        if (!has_explicit_source_map && ends_with(source_name, ".weight")) {
            const auto base = source_name.substr(0, source_name.size() - 7);
            candidates.push_back(source_name + "_scale_inv");
            candidates.push_back(source_name + "_scale");
            candidates.push_back(base + ".weight_scale_inv");
            candidates.push_back(base + ".scale");
            candidates.push_back(base + ".weight_scale");
            candidates.push_back(base + ".weight_scale_2");
        }
        return candidates;
    };
    const auto find_scale = [&](const std::string& source_name,
                                const std::string& canonical_name,
                                const auto& accepted) {
        for (const auto& candidate :
             scale_candidates(source_name, canonical_name)) {
            const auto* tensor = impl_->source.find_tensor(candidate);
            if (tensor != nullptr && accepted(tensor->dtype)) {
                return candidate;
            }
        }
        return std::string{};
    };

    std::unordered_set<std::string> consumed_scales;
    std::vector<std::pair<std::string, std::string>> aliases(
        canonical_to_source.begin(), canonical_to_source.end());
    std::sort(aliases.begin(), aliases.end());
    for (const auto& [canonical, source_name] : aliases) {
        const auto* values = impl_->source.find_tensor(source_name);
        if (values == nullptr || values->dtype == "F8_E8M0") continue;
        if (!values->shape) {
            throw std::runtime_error("HF tensor shape is missing: " + source_name);
        }

        Impl::Record record;
        record.metadata.name = canonical;
        record.metadata.stored_dtype = values->dtype;
        record.metadata.shape = values->shape;
        const auto item_size = dense_item_size(values->dtype);
        if (item_size != 0) {
            if (checked_product(*values->shape, item_size, source_name) !=
                values->nbytes) {
                throw std::runtime_error(
                    "dense Safetensors byte size mismatch: " + source_name);
            }
            record.metadata.dtype = values->dtype;
            auto prefix = dense_prefix(*values->shape);
            record.segments.push_back({0, std::move(prefix), {}, 0, 0});
            const auto values_offset = record.segments.front().inline_bytes.size();
            record.segments.push_back({
                values_offset, {}, source_name, 0, values->nbytes});
            record.metadata.nbytes = checked_add(
                values_offset, values->nbytes, "dense tensor");
        } else if (values->dtype == "I8" || values->dtype == "F8_E4M3" ||
                   values->dtype == "F8_E4M3FN") {
            if (checked_product(*values->shape, 1, source_name) !=
                values->nbytes) {
                throw std::runtime_error(
                    "native FP8/MX values byte size mismatch: " + source_name);
            }
            auto logical_shape = *values->shape;
            std::string scale_name;
            if (values->dtype != "I8") {
                scale_name = find_scale(
                    source_name, canonical, [](std::string_view dtype) {
                        return dtype == "BF16" || dtype == "F16" ||
                            dtype == "F32";
                    });
            }
            if (!scale_name.empty()) {
                record.metadata.dtype = "FP8-128SQ";
            } else {
                scale_name = find_scale(
                    source_name, canonical, [](std::string_view dtype) {
                        return dtype == "F8_E8M0";
                    });
                if (scale_name.empty()) {
                    throw std::runtime_error(
                        "native FP8/MX tensor has no supported scale binding: " +
                        source_name);
                }
                if (values->dtype == "I8") {
                    if (logical_shape.size() != 2 || logical_shape[1] >
                            std::numeric_limits<std::int64_t>::max() / 2) {
                        throw std::runtime_error(
                            "invalid native MXFP4 tensor shape: " + source_name);
                    }
                    logical_shape[1] *= 2;
                    record.metadata.dtype = "MXFP4";
                } else {
                    record.metadata.dtype = "MXFP8";
                }
            }
            record.metadata.stored_dtype = record.metadata.dtype;
            record.metadata.shape = logical_shape;
            const auto* scales = impl_->source.find_tensor(scale_name);
            if (scales == nullptr || !scales->shape) {
                throw std::runtime_error("native FP8/MX scale shape is missing: " +
                                         scale_name);
            }
            const auto scale_item_size = dense_item_size(scales->dtype);
            const bool block_fp8 = record.metadata.dtype == "FP8-128SQ";
            const auto expected_scale_bytes = block_fp8
                ? checked_product(*scales->shape, scale_item_size, scale_name)
                : checked_product(*scales->shape, 1, scale_name);
            if ((block_fp8 && scale_item_size == 0) ||
                expected_scale_bytes != scales->nbytes) {
                throw std::runtime_error(
                    "native FP8/MX scale byte size mismatch: " + scale_name);
            }
            auto prefix = record.metadata.dtype == "FP8-128SQ"
                ? fp8_128_sq8_prefix(
                    logical_shape, scales->dtype, *scales->shape, source_name)
                : mx_prefix(record.metadata.dtype, logical_shape,
                            *values->shape, *scales->shape, source_name);
            const auto values_offset = prefix.size();
            const auto scales_offset = checked_add(
                values_offset, values->nbytes, "native FP8/MX tensor");
            record.segments.push_back({0, std::move(prefix), {}, 0, 0});
            record.segments.push_back({
                values_offset, {}, source_name, 0, values->nbytes});
            record.segments.push_back({
                scales_offset, {}, scale_name, 0, scales->nbytes});
            record.metadata.nbytes = checked_add(
                scales_offset, scales->nbytes, "native FP8/MX tensor");
            consumed_scales.insert(scale_name);
        } else {
            throw std::runtime_error("unsupported full-precision HF dtype " +
                                     values->dtype + ": " + source_name);
        }
        if (!impl_->records.emplace(canonical, std::move(record)).second) {
            throw std::runtime_error("duplicate canonical HF tensor: " + canonical);
        }
    }
    for (const auto& scale : consumed_scales) {
        const auto canonical = source_to_canonical.find(scale);
        if (canonical != source_to_canonical.end()) {
            impl_->records.erase(canonical->second);
        }
    }
    for (const auto& tensor : impl_->source.tensors()) {
        if (tensor.dtype == "F8_E8M0" &&
            source_to_canonical.find(tensor.name) != source_to_canonical.end() &&
            consumed_scales.find(tensor.name) == consumed_scales.end()) {
            throw std::runtime_error("orphan E8M0 scale tensor: " + tensor.name);
        }
    }

    struct Projection {
        std::map<std::size_t, std::string> experts;
    };
    std::map<std::string, Projection> projections;
    const auto collect = [&](std::string_view canonical) {
        constexpr std::string_view marker = ".mlp.experts.";
        const auto marker_offset = canonical.find(marker);
        if (marker_offset == std::string_view::npos) return;
        const auto expert_begin = marker_offset + marker.size();
        const auto expert_end = canonical.find('.', expert_begin);
        if (expert_end == std::string_view::npos || expert_end == expert_begin) return;
        std::size_t expert = 0;
        const auto id = canonical.substr(expert_begin, expert_end - expert_begin);
        const auto parsed = std::from_chars(
            id.data(), id.data() + id.size(), expert);
        if (parsed.ec != std::errc{} || parsed.ptr != id.data() + id.size()) return;
        const auto projection_end = canonical.find('.', expert_end + 1);
        if (projection_end == std::string_view::npos ||
            canonical.substr(projection_end) != ".weight") return;
        const auto projection = canonical.substr(
            expert_end + 1, projection_end - expert_end - 1);
        if (projection != "gate" && projection != "up" && projection != "down") {
            return;
        }
        const auto aggregate = std::string(canonical.substr(0, marker_offset)) +
            std::string(marker) + std::string(projection) + ".weight";
        projections[aggregate].experts.emplace(expert, std::string(canonical));
    };
    for (const auto& [name, unused] : impl_->records) {
        static_cast<void>(unused);
        collect(name);
    }

    for (const auto& [name, projection] : projections) {
        if (impl_->records.find(name) != impl_->records.end() ||
            projection.experts.empty()) continue;
        const auto expert_count = projection.experts.rbegin()->first + 1;
        if (projection.experts.size() != expert_count) {
            throw std::runtime_error(
                "native HF expert projection has missing expert IDs: " + name);
        }
        Impl::Record aggregate;
        aggregate.metadata.name = name;
        aggregate.metadata.dtype = "MFE";
        aggregate.metadata.stored_dtype = "MFE";
        std::string dtype;
        std::vector<std::int64_t> shape;
        std::uint64_t logical_offset = 0;
        const auto append_inline = [&](std::vector<std::byte> bytes) {
            const auto count = bytes.size();
            aggregate.segments.push_back({
                logical_offset, std::move(bytes), {}, 0, count});
            logical_offset = checked_add(
                logical_offset, count, "virtual MFE metadata");
        };
        const auto append_source_segment = [&](const Impl::Segment& segment) {
            if (!segment.inline_bytes.empty()) {
                append_inline(segment.inline_bytes);
            } else {
                aggregate.segments.push_back({
                    logical_offset, {}, segment.source_name,
                    segment.source_offset, segment.nbytes});
                logical_offset = checked_add(
                    logical_offset, segment.nbytes, "virtual MFE tensor");
            }
        };

        std::vector<std::byte> header;
        for (const char value : std::string_view("MFE1")) {
            header.push_back(static_cast<std::byte>(value));
        }
        append_little<std::uint32_t>(
            header, static_cast<std::uint32_t>(expert_count));
        append_little<std::uint32_t>(header, 0);
        append_little<std::uint32_t>(header, 0);
        append_little<std::uint32_t>(
            header, static_cast<std::uint32_t>(expert_count));

        for (const auto& [expert, tensor_name] : projection.experts) {
            const auto& record = impl_->records.at(tensor_name);
            if (!record.metadata.shape || record.metadata.shape->size() != 2 ||
                record.segments.empty()) {
                throw std::runtime_error(
                    "native HF expert must be rank two: " + tensor_name);
            }
            if (dtype.empty()) {
                dtype = record.metadata.dtype;
                shape = *record.metadata.shape;
                if (shape[0] <= 0 || shape[1] <= 0 ||
                    shape[0] > std::numeric_limits<std::uint32_t>::max() ||
                    shape[1] > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error(
                        "native HF expert geometry exceeds MFE: " + tensor_name);
                }
                const auto output = static_cast<std::uint32_t>(shape[0]);
                const auto input = static_cast<std::uint32_t>(shape[1]);
                std::memcpy(header.data() + 8, &output, sizeof(output));
                std::memcpy(header.data() + 12, &input, sizeof(input));
                append_inline(std::move(header));
            } else if (dtype != record.metadata.dtype ||
                       shape != *record.metadata.shape) {
                throw std::runtime_error(
                    "native HF expert projection has mixed geometry: " + name);
            }

            std::vector<std::byte> pool;
            append_little<std::uint32_t>(pool, 1);
            append_little<std::uint32_t>(
                pool, static_cast<std::uint32_t>(dtype.size()));
            append_little<std::uint64_t>(pool, record.metadata.nbytes);
            append_little<std::uint64_t>(pool, 0);
            append_little<std::int32_t>(pool, static_cast<std::int32_t>(expert));
            for (const char value : dtype) {
                pool.push_back(static_cast<std::byte>(value));
            }
            pool.insert(pool.end(), record.segments.front().inline_bytes.begin(),
                        record.segments.front().inline_bytes.end());
            append_inline(std::move(pool));
            for (std::size_t index = 1; index < record.segments.size(); ++index) {
                append_source_segment(record.segments[index]);
            }
        }
        aggregate.metadata.nbytes = logical_offset;
        impl_->records.emplace(name, std::move(aggregate));
    }

    if (impl_->records.size() > kMaxRecords) {
        throw std::runtime_error("HF canonical tensor count exceeds the record limit");
    }
    std::vector<std::string> names;
    names.reserve(impl_->records.size());
    for (const auto& [name, unused] : impl_->records) {
        static_cast<void>(unused);
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    for (const auto& name : names) {
        impl_->tensor_indices.emplace(name, impl_->tensors.size());
        impl_->tensors.push_back(impl_->records.at(name).metadata);
    }
}

HfModelSource::~HfModelSource() = default;
HfModelSource::HfModelSource(HfModelSource&&) noexcept = default;
HfModelSource& HfModelSource::operator=(HfModelSource&&) noexcept = default;

const std::vector<std::filesystem::path>& HfModelSource::source_paths() const noexcept {
    return impl_->source.source_paths();
}
void HfModelSource::drop_file_cache() const noexcept {
    impl_->source.drop_file_cache();
}
std::string_view HfModelSource::architecture() const noexcept {
    return impl_->source.architecture();
}
const std::unordered_map<std::string, std::string>&
HfModelSource::metadata() const noexcept {
    return impl_->source.metadata();
}
const std::vector<TensorMetadata>& HfModelSource::tensors() const noexcept {
    return impl_->tensors;
}
const TensorMetadata* HfModelSource::find_tensor(
    std::string_view name) const noexcept {
    const auto found = impl_->tensor_indices.find(std::string(name));
    return found == impl_->tensor_indices.end()
        ? nullptr
        : &impl_->tensors[found->second];
}

void HfModelSource::read_range_into(
    std::string_view name,
    std::uint64_t relative_offset,
    std::byte* destination,
    std::size_t size) const {
    const auto found = impl_->records.find(std::string(name));
    if (found == impl_->records.end()) {
        throw std::runtime_error("model tensor not found: " + std::string(name));
    }
    const auto& record = found->second;
    if (relative_offset > record.metadata.nbytes ||
        size > record.metadata.nbytes - relative_offset) {
        throw std::out_of_range("model tensor byte range is out of bounds: " +
                                std::string(name));
    }
    const auto request_end = relative_offset + size;
    for (const auto& segment : record.segments) {
        const auto segment_end = checked_add(
            segment.offset,
            segment.inline_bytes.empty() ? segment.nbytes
                                         : segment.inline_bytes.size(),
            "HF virtual tensor segment");
        const auto begin = std::max(relative_offset, segment.offset);
        const auto end = std::min<std::uint64_t>(request_end, segment_end);
        if (begin >= end) continue;
        const auto target_offset = static_cast<std::size_t>(begin - relative_offset);
        const auto count = static_cast<std::size_t>(end - begin);
        if (!segment.inline_bytes.empty()) {
            std::copy_n(
                segment.inline_bytes.data() + (begin - segment.offset), count,
                destination + target_offset);
        } else {
            impl_->source.read_range_into(
                segment.source_name,
                segment.source_offset + begin - segment.offset,
                destination + target_offset,
                count);
        }
    }
}

const std::vector<std::string>& HfModelSource::assets() const noexcept {
    return impl_->source.assets();
}
bool HfModelSource::has_asset(std::string_view name) const noexcept {
    return impl_->source.has_asset(name);
}
std::vector<std::byte> HfModelSource::read_asset(std::string_view name) const {
    return impl_->source.read_asset(name);
}
std::optional<ModelGraph> HfModelSource::model_graph() const {
    return impl_->source.model_graph();
}

} // namespace mfq
