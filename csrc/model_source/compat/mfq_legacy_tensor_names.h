#pragma once

#include "tensor_layout.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mfq {

struct MfqLegacyTensorAliases {
    std::unordered_map<std::string, std::string> canonical_to_stored;
    TensorLayout layout;
};

// The sole backend-neutral C++ name compatibility boundary. Runtime loaders
// ask only for canonical names; this view aliases them to records from older
// MFQ files without copying tensor blobs.
MfqLegacyTensorAliases make_legacy_tensor_aliases(
    std::string_view artifact_architecture,
    std::string_view model_config_json,
    const std::vector<std::string>& stored_names);

} // namespace mfq
