#pragma once

#include "mfq/model_source.h"

#include <filesystem>
#include <memory>
#include <unordered_map>

namespace mfq {

struct HfTensorLocation {
    std::size_t shard = 0;
    std::uint64_t offset = 0;
};

class HfSafetensorsSource final : public ModelSource {
public:
    explicit HfSafetensorsSource(
        std::filesystem::path root,
        std::unordered_map<std::string, std::string> canonical_to_source = {});
    ~HfSafetensorsSource() override;

    HfSafetensorsSource(HfSafetensorsSource&&) noexcept;
    HfSafetensorsSource& operator=(HfSafetensorsSource&&) noexcept;
    HfSafetensorsSource(const HfSafetensorsSource&) = delete;
    HfSafetensorsSource& operator=(const HfSafetensorsSource&) = delete;

    const std::filesystem::path& root() const noexcept;
    const std::vector<std::filesystem::path>& source_paths()
        const noexcept override;
    const HfTensorLocation& location(std::string_view name) const;
    void drop_file_cache() const noexcept override;

    std::string_view architecture() const noexcept override;
    const std::unordered_map<std::string, std::string>& metadata()
        const noexcept override;
    const std::vector<TensorMetadata>& tensors() const noexcept override;
    const TensorMetadata* find_tensor(
        std::string_view name) const noexcept override;
    void read_range_into(
        std::string_view name,
        std::uint64_t relative_offset,
        std::byte* destination,
        std::size_t size) const override;
    const std::vector<std::string>& assets() const noexcept override;
    bool has_asset(std::string_view name) const noexcept override;
    std::vector<std::byte> read_asset(std::string_view name) const override;
    std::optional<ModelGraph> model_graph() const override;

    void read_shard_range_into(
        std::size_t shard,
        std::uint64_t offset,
        std::byte* destination,
        std::size_t size) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq
