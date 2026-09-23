#pragma once

#include "mfq/model_graph.h"
#include "mfq_legacy_tensor_names.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mfq {

inline constexpr std::string_view kModelConfigAsset =
    "__mfq_asset__/model_config.json";

struct TensorMetadata {
    std::string name;
    std::string dtype;
    std::string stored_dtype;
    std::optional<std::vector<std::int64_t>> shape;
    std::uint64_t nbytes = 0;
};

// Backend-neutral, read-only model input. Implementations normalize storage
// names before tensors cross this boundary; CUDA and Metal only consume the
// canonical names exposed here.
class ModelSource {
public:
    virtual ~ModelSource() = default;

    virtual const std::vector<std::filesystem::path>& source_paths()
        const noexcept = 0;
    virtual std::string_view architecture() const noexcept = 0;
    virtual const std::unordered_map<std::string, std::string>& metadata()
        const noexcept = 0;
    virtual const std::vector<TensorMetadata>& tensors() const noexcept = 0;
    virtual const TensorMetadata* find_tensor(
        std::string_view name) const noexcept = 0;
    virtual const MfqLegacyTensorAliases& legacy_tensor_compatibility()
        const noexcept {
        static const MfqLegacyTensorAliases compatibility;
        return compatibility;
    }
    virtual void read_range_into(
        std::string_view name,
        std::uint64_t relative_offset,
        std::byte* destination,
        std::size_t size) const = 0;
    virtual void drop_file_cache() const noexcept {}

    virtual const std::vector<std::string>& assets() const noexcept = 0;
    virtual bool has_asset(std::string_view name) const noexcept = 0;
    virtual std::vector<std::byte> read_asset(std::string_view name) const = 0;
    // The stored graph may be absent in legacy artifacts. Runtime callers
    // should use resolved_model_graph(), which applies compatibility here at
    // the source boundary.
    virtual std::optional<ModelGraph> model_graph() const = 0;

    std::string model_config_json() const;
    ModelGraph resolved_model_graph() const;

    std::vector<std::byte> read(
        std::string_view name,
        std::uint64_t relative_offset = 0,
        std::optional<std::size_t> size = std::nullopt) const {
        const auto* tensor = find_tensor(name);
        if (tensor == nullptr) {
            throw std::runtime_error("model tensor not found: " + std::string(name));
        }
        if (relative_offset > tensor->nbytes) {
            throw std::out_of_range("model tensor byte range is out of bounds: " +
                                    std::string(name));
        }
        const auto available = tensor->nbytes - relative_offset;
        if (!size && available > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error("model tensor is too large to read");
        }
        const auto requested = size ? *size : static_cast<std::size_t>(available);
        if (requested > available) {
            throw std::out_of_range("model tensor byte range is out of bounds: " +
                                    std::string(name));
        }
        std::vector<std::byte> result(requested);
        read_range_into(name, relative_offset, result.data(), result.size());
        return result;
    }
};

std::shared_ptr<const ModelSource> open_model_source(
    const std::filesystem::path& path);

} // namespace mfq
