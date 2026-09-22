#include "mfq/fp8_sq_blob.h"
#include "mfq/hf_model_source.h"
#include "mfq/hf_safetensors_source.h"
#include "mfq/mfq_model_source.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void write_scalar(std::ofstream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_string(std::ofstream& stream, std::string_view value) {
    write_scalar(stream, static_cast<std::uint32_t>(value.size()));
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

std::string graph_json() {
    return R"json({
        "schema_version": 1,
        "architecture": "qwen3_5",
        "canonical_naming": {
            "namespace": "mfq.tensor",
            "version": 1,
            "component_roots": ["model"]
        },
        "topology": {"text_layers": 1, "vision_layers": 0, "predictor_layers": 0},
        "graph": {
            "kind": "causal_lm",
            "backbone": "qwen3_5",
            "components": [{
                "kind": "text",
                "tensor_root": "model",
                "implementation": "qwen3_5"
            }]
        },
        "capabilities": ["text"]
    })json";
}

void write_mfq(const std::filesystem::path& path) {
    const std::string config = R"({"model_type":"qwen3_5"})";
    std::ofstream stream(path, std::ios::binary);
    stream.write("MFQ1", 4);
    write_scalar<std::uint32_t>(stream, 2);
    write_string(stream, "qwen3_5");
    write_scalar<std::uint32_t>(stream, 0);
    write_scalar<std::uint32_t>(stream, 2);
    write_string(stream, "model.token_embedding.weight");
    write_string(stream, "BF16");
    write_scalar<std::uint64_t>(stream, 14);
    write_string(stream, "__mfq_asset__/model_config.json");
    write_string(stream, "BLOB");
    write_scalar<std::uint64_t>(stream, config.size());
    write_scalar<std::uint32_t>(stream, 1);
    write_scalar<std::int64_t>(stream, 1);
    const char tensor[] = {0x12, 0x34};
    stream.write(tensor, sizeof(tensor));
    stream.write(config.data(), static_cast<std::streamsize>(config.size()));
}

void write_hf(const std::filesystem::path& root) {
    std::filesystem::create_directories(root / ".mfq-assets" / "hf");
    std::ofstream(root / "config.json") << R"({"model_type":"qwen3_5"})";
    std::ofstream(root / ".mfq-assets" / "model_graph.json") << graph_json();
    std::ofstream(root / ".mfq-assets" / "hf" / "source_tensor_map.json")
        << R"({"schema":"mfq.hf-source-map","version":1,"canonical_to_source":{"model.token_embedding.weight":"raw.embed.weight","model.blocks.0.mlp.experts.0.gate.weight":"raw.e0.weight","model.blocks.0.mlp.experts.0.gate.weight_scale":"raw.e0.scale","model.blocks.0.mlp.experts.1.gate.weight":"raw.e1.weight","model.blocks.0.mlp.experts.1.gate.weight_scale":"raw.e1.scale"}})";
    const std::string header =
        R"({"raw.embed.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"raw.e0.weight":{"dtype":"I8","shape":[1,16],"data_offsets":[2,18]},"raw.e0.scale":{"dtype":"F8_E8M0","shape":[1,1],"data_offsets":[18,19]},"raw.e1.weight":{"dtype":"I8","shape":[1,16],"data_offsets":[19,35]},"raw.e1.scale":{"dtype":"F8_E8M0","shape":[1,1],"data_offsets":[35,36]}})";
    std::ofstream stream(root / "model.safetensors", std::ios::binary);
    write_scalar<std::uint64_t>(stream, header.size());
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    const char tensor[] = {0x12, 0x34};
    stream.write(tensor, sizeof(tensor));
    std::array<char, 34> experts{};
    experts[0] = 0x45;
    experts[16] = 0x7f;
    experts[17] = 0x46;
    experts[33] = 0x7e;
    stream.write(experts.data(), experts.size());
}

void write_hf_mxfp8(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"qwen3_5"})";
    const std::string header =
        R"({"block128.weight":{"dtype":"F8_E4M3","shape":[128,128],"data_offsets":[0,16384]},"block128.scale":{"dtype":"F8_E8M0","shape":[1,1],"data_offsets":[16384,16385]},"block32.weight":{"dtype":"F8_E4M3","shape":[33,64],"data_offsets":[16385,18497]},"block32.scale":{"dtype":"F8_E8M0","shape":[2,2],"data_offsets":[18497,18501]},"row32.weight":{"dtype":"F8_E4M3","shape":[3,64],"data_offsets":[18501,18693]},"row32.scale":{"dtype":"F8_E8M0","shape":[3,2],"data_offsets":[18693,18699]}})";
    std::ofstream stream(root / "model.safetensors", std::ios::binary);
    write_scalar<std::uint64_t>(stream, header.size());
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::vector<char> payload(18699);
    stream.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

void write_hf_fp8_128(
    const std::filesystem::path& root,
    bool explicit_source_map = false) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"qwen3_5"})";
    if (explicit_source_map) {
        std::filesystem::create_directories(root / ".mfq-assets" / "hf");
        std::ofstream(root / ".mfq-assets" / "hf" / "source_tensor_map.json")
            << R"({"schema":"mfq.hf-source-map","version":1,"canonical_to_source":{"model.block.0.linear_attention.gate.weight":"block.weight","model.block.0.linear_attention.gate.weight_scale":"block.weight_scale_inv"}})";
    }
    const std::string header =
        R"({"block.weight":{"dtype":"F8_E4M3","shape":[128,128],"data_offsets":[0,16384]},"block.weight_scale_inv":{"dtype":"BF16","shape":[1,1],"data_offsets":[16384,16386]}})";
    std::ofstream stream(root / "model.safetensors", std::ios::binary);
    write_scalar<std::uint64_t>(stream, header.size());
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    std::array<char, 16384> values{};
    values.front() = 0x12;
    values.back() = 0x34;
    stream.write(values.data(), static_cast<std::streamsize>(values.size()));
    const std::array<char, 2> bf16_one = {
        static_cast<char>(0x80), static_cast<char>(0x3f)};
    stream.write(bf16_one.data(), static_cast<std::streamsize>(bf16_one.size()));
}

void require_bytes(const std::vector<std::byte>& bytes) {
    require(bytes.size() == 14, "unexpected tensor byte count");
    require(std::to_integer<unsigned char>(bytes[12]) == 0x12 &&
                std::to_integer<unsigned char>(bytes[13]) == 0x34,
            "unexpected tensor bytes");
}

} // namespace

int main() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() /
        ("mfq-model-source-test-" + std::to_string(nonce));
    try {
        std::filesystem::create_directories(root);
        const auto mfq_path = root / "model.mfq";
        const auto hf_path = root / "hf";
        write_mfq(mfq_path);
        write_hf(hf_path);
        write_hf_mxfp8(root / "hf-mxfp8");
        write_hf_fp8_128(root / "hf-fp8-128");
        write_hf_fp8_128(root / "hf-fp8-128-explicit", true);

        mfq::MfqModelSource mfq_source(mfq_path);
        mfq::HfModelSource hf_source(hf_path);
        const auto opened_mfq = mfq::open_model_source(mfq_path);
        const auto opened_hf = mfq::open_model_source(hf_path);

        require(mfq_source.tensors().size() == 1,
                "MFQ assets leaked into tensor enumeration");

        for (const mfq::ModelSource* source :
             std::vector<const mfq::ModelSource*>{
                 opened_mfq.get(), opened_hf.get()}) {
            const auto* tensor = source->find_tensor(
                "model.token_embedding.weight");
            require(tensor != nullptr, "canonical tensor is missing");
            require(tensor->dtype == "BF16" && tensor->nbytes == 14,
                    "canonical tensor metadata differs by source format");
            require_bytes(source->read("model.token_embedding.weight"));
            std::byte ignored{};
            source->read_range_into(
                "model.token_embedding.weight", 2, &ignored, 0);
            require(source->resolved_model_graph().backbone == "qwen3_5",
                    "model graph differs by source format");
            require(source->model_config_json() ==
                        R"({"model_type":"qwen3_5"})",
                    "model config differs by source format");
        }
        require(!mfq_source.find_tensor("__mfq_asset__/model_graph.json"),
                "MFQ graph asset is exposed as a tensor");
        require(!mfq_source.has_asset("__mfq_asset__/model_graph.json"),
                "legacy MFQ unexpectedly has a stored graph");
        require(mfq_source.has_asset("__mfq_asset__/model_config.json"),
                "MFQ config asset is missing");
        require(hf_source.has_asset("__mfq_asset__/model_graph.json"),
                "HF graph asset is missing");
        require(hf_source.find_tensor("model.token_embedding.weight")->shape ==
                    std::optional<std::vector<std::int64_t>>({1}),
                "HF tensor shape was lost");
        const auto dense_cross = hf_source.read(
            "model.token_embedding.weight", 11, 3);
        require(dense_cross.size() == 3 &&
                    std::to_integer<unsigned char>(dense_cross[1]) == 0x12 &&
                    std::to_integer<unsigned char>(dense_cross[2]) == 0x34,
                "dense segmented range read is invalid");
        const auto* mx = hf_source.find_tensor(
            "model.blocks.0.mlp.experts.0.gate.weight");
        require(mx != nullptr && mx->dtype == "MXFP4" && mx->nbytes == 73,
                "native MX view was not synthesized");
        const auto mx_bytes = hf_source.read(mx->name);
        require(mx_bytes.size() == 73 &&
                    std::to_integer<char>(mx_bytes[0]) == 'M' &&
                    std::to_integer<char>(mx_bytes[3]) == '1' &&
                    std::to_integer<unsigned char>(mx_bytes[56]) == 0x45 &&
                    std::to_integer<unsigned char>(mx_bytes.back()) == 0x7f,
                "native MX payload is invalid");
        const auto mx_cross = hf_source.read(mx->name, 55, 2);
        require(mx_cross.size() == 2 &&
                    std::to_integer<unsigned char>(mx_cross[1]) == 0x45,
                "native MX segmented range read is invalid");
        const auto* mfe = hf_source.find_tensor(
            "model.blocks.0.mlp.experts.gate.weight");
        require(mfe != nullptr && mfe->dtype == "MFE" && mfe->nbytes == 232,
                "virtual MFE view was not synthesized");
        const auto mfe_bytes = hf_source.read(mfe->name);
        require(mfe_bytes.size() == 232 &&
                    std::to_integer<char>(mfe_bytes[0]) == 'M' &&
                    std::to_integer<char>(mfe_bytes[3]) == '1',
                "virtual MFE payload is invalid");

        mfq::HfModelSource fp8_128_source(root / "hf-fp8-128");
        const auto* fp8_128 = fp8_128_source.find_tensor("block.weight");
        require(fp8_128 != nullptr && fp8_128->dtype == "FP8-128SQ" &&
                    fp8_128->stored_dtype == "FP8-128SQ" &&
                    fp8_128->nbytes == 16734,
                "native block FP8 was not exposed as lossless FP8-128SQ");
        require(fp8_128_source.find_tensor("block.weight_scale_inv") == nullptr,
                "consumed block FP8 scale leaked into tensor enumeration");
        const auto fp8_128_bytes = fp8_128_source.read(fp8_128->name);
        const auto fp8_128_layout = mfq::fp8sq::parse(
            fp8_128->dtype,
            reinterpret_cast<const std::uint8_t*>(fp8_128_bytes.data()),
            fp8_128_bytes.size());
        const auto fp8_128_rows = mfq::fp8sq::row_metadata(
            reinterpret_cast<const std::uint8_t*>(fp8_128_bytes.data()),
            fp8_128_layout);
        require(fp8_128_layout.outputs == 128 &&
                    fp8_128_layout.width == 128 &&
                    fp8_128_layout.block_rows == 128 &&
                    fp8_128_layout.block_columns == 128 &&
                    fp8_128_layout.scale_kind == mfq::fp8sq::ScaleKind::Bf16 &&
                    std::all_of(
                        fp8_128_rows.q.begin(), fp8_128_rows.q.end(),
                        [](std::uint8_t q) { return q == 8; }) &&
                    std::to_integer<unsigned char>(
                        fp8_128_bytes[fp8_128_layout.symbols]) == 0x12 &&
                    std::to_integer<unsigned char>(
                        fp8_128_bytes[fp8_128_layout.scales - 1]) == 0x34 &&
                    std::to_integer<unsigned char>(
                        fp8_128_bytes[fp8_128_layout.scales]) == 0x80 &&
                    std::to_integer<unsigned char>(
                        fp8_128_bytes[fp8_128_layout.scales + 1]) == 0x3f,
                "lossless FP8-128SQ payload differs from HF source bytes");

        mfq::HfModelSource fp8_128_explicit_source(
            root / "hf-fp8-128-explicit");
        const auto* fp8_128_explicit = fp8_128_explicit_source.find_tensor(
            "model.block.0.linear_attention.gate.weight");
        require(fp8_128_explicit != nullptr &&
                    fp8_128_explicit->dtype == "FP8-128SQ" &&
                    fp8_128_explicit->stored_dtype == "FP8-128SQ" &&
                    fp8_128_explicit->nbytes == 16734,
                "explicit HF source map did not bind block FP8 scale");
        require(fp8_128_explicit_source.find_tensor(
                    "model.block.0.linear_attention.gate.weight_scale") == nullptr,
                "explicit block FP8 scale leaked into tensor enumeration");

        mfq::HfModelSource mxfp8_source(root / "hf-mxfp8");
        for (const auto& [name, nbytes] :
             std::array<std::pair<const char*, std::uint64_t>, 3>{{
                 {"block128.weight", 16441},
                 {"block32.weight", 2172},
                 {"row32.weight", 254},
             }}) {
            const auto* tensor = mxfp8_source.find_tensor(name);
            require(tensor != nullptr && tensor->dtype == "MXFP8" &&
                        tensor->nbytes == nbytes,
                    "valid native MXFP8 geometry was rejected");
        }

        bool rejected = false;
        try {
            static_cast<void>(hf_source.read(
                "model.token_embedding.weight", 13, 2));
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        require(rejected, "out-of-range tensor read was accepted");

        std::filesystem::remove_all(root);
        std::cout << "model source parity passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << "model source parity failed: " << error.what() << '\n';
        return 1;
    }
}
