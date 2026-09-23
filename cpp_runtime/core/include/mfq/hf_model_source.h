#pragma once

#include "mfq/hf_safetensors_source.h"

#include <filesystem>
#include <memory>

namespace mfq {

// Canonical runtime view of an HF checkpoint. Dense and native-MX tensors are
// exposed with the same logical payload contract as MFQ records, so backend
// model loaders do not branch on checkpoint format.
class HfModelSource final : public ModelSource {
public:
    explicit HfModelSource(std::filesystem::path root);
    ~HfModelSource() override;

    HfModelSource(HfModelSource&&) noexcept;
    HfModelSource& operator=(HfModelSource&&) noexcept;
    HfModelSource(const HfModelSource&) = delete;
    HfModelSource& operator=(const HfModelSource&) = delete;

    const std::vector<std::filesystem::path>& source_paths()
        const noexcept override;
    void drop_file_cache() const noexcept override;

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
    const std::vector<std::string>& assets() const noexcept override;
    bool has_asset(std::string_view name) const noexcept override;
    std::vector<std::byte> read_asset(std::string_view name) const override;
    std::optional<ModelGraph> model_graph() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq
