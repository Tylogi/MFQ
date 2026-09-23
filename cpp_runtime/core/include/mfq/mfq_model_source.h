#pragma once

#include "mfq/model_source.h"

#include <filesystem>
#include <memory>
#include <unordered_map>

namespace mfq {

struct MfqSourceHeader {
    std::uint32_t version = 0;
    std::string architecture;
    std::unordered_map<std::string, std::string> metadata;
    std::uint32_t record_count = 0;
};

struct MfqStoredRecord {
    TensorMetadata tensor;
    std::filesystem::path source_path;
    std::uint64_t offset = 0;
    bool asset = false;
};

class MfqModelSource final : public ModelSource {
public:
    explicit MfqModelSource(std::filesystem::path path);
    ~MfqModelSource() override;

    MfqModelSource(MfqModelSource&&) noexcept;
    MfqModelSource& operator=(MfqModelSource&&) noexcept;
    MfqModelSource(const MfqModelSource&) = delete;
    MfqModelSource& operator=(const MfqModelSource&) = delete;

    const MfqSourceHeader& header() const noexcept;
    const std::vector<std::filesystem::path>& source_paths()
        const noexcept override;
    const std::vector<MfqStoredRecord>& records() const noexcept;

    std::string_view architecture() const noexcept override;
    const std::unordered_map<std::string, std::string>& metadata()
        const noexcept override;
    const std::vector<TensorMetadata>& tensors() const noexcept override;
    const TensorMetadata* find_tensor(
        std::string_view name) const noexcept override;
    const MfqLegacyTensorAliases& legacy_tensor_compatibility()
        const noexcept override;
    void read_range_into(
        std::string_view name,
        std::uint64_t relative_offset,
        std::byte* destination,
        std::size_t size) const override;
    void drop_file_cache() const noexcept override;
    const std::vector<std::string>& assets() const noexcept override;
    bool has_asset(std::string_view name) const noexcept override;
    std::vector<std::byte> read_asset(std::string_view name) const override;
    std::optional<ModelGraph> model_graph() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq
