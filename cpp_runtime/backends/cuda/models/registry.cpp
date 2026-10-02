#include "storage/weight_loader.h"
#include "registry.h"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>

namespace mfq::cuda {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open config: " + path);
    return {std::istreambuf_iterator<char>(input), {}};
}

bool has_tensor_prefix(
        const mfq::ModelSource& source,
        std::string_view prefix) {
    for (const auto& tensor : source.tensors()) {
        if (tensor.name == prefix ||
                (tensor.name.size() > prefix.size() &&
                 tensor.name.compare(0, prefix.size(), prefix) == 0 &&
                 tensor.name[prefix.size()] == '.')) {
            return true;
        }
    }
    return false;
}

} // namespace

std::string load_model_config_json(
        const mfq::ModelSource& source,
        const std::string& external_path) {
    return external_path.empty()
        ? source.model_config_json()
        : read_file(external_path);
}

void validate_model_source(const mfq::ModelSource& source) {
    if (source.find_tensor("model.token_embedding.weight") == nullptr) {
        throw std::runtime_error(
            "canonical text tensor inventory has no token embedding");
    }
    for (const auto& component : source.resolved_model_graph().components) {
        if (component.tensor_root != "runtime" &&
                !has_tensor_prefix(source, component.tensor_root)) {
            throw std::runtime_error(
                "model graph declares " + component.kind +
                " but its canonical tensor inventory is missing");
        }
    }
}

} // namespace mfq::cuda
