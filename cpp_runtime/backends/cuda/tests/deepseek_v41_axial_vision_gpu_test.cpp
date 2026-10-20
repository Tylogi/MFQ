#include "core/axial_patch_vision.h"

#include "mfq/model_source.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::string model;
    std::string patches;
    std::string output;
    int height = 0;
    int width = 0;
    int hidden = 0;
    int layers = 0;
    int heads = 0;
    int intermediate = 0;
    int patch_size = 0;
    int downsample_ratio = 0;
    int output_hidden = 0;
};

std::string required(int& index, int argc, char** argv) {
    if (++index >= argc) {
        throw std::invalid_argument(
            std::string("missing value after ") + argv[index - 1]);
    }
    return argv[index];
}

int positive(const std::string& value, const char* name) {
    const int parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::invalid_argument(
            std::string("invalid axial vision ") + name);
    }
    return parsed;
}

Arguments parse(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model") {
            result.model = required(index, argc, argv);
        } else if (argument == "--patches") {
            result.patches = required(index, argc, argv);
        } else if (argument == "--output") {
            result.output = required(index, argc, argv);
        } else if (argument == "--height") {
            result.height = positive(required(index, argc, argv), "height");
        } else if (argument == "--width") {
            result.width = positive(required(index, argc, argv), "width");
        } else if (argument == "--hidden") {
            result.hidden = positive(required(index, argc, argv), "hidden");
        } else if (argument == "--layers") {
            result.layers = positive(required(index, argc, argv), "layers");
        } else if (argument == "--heads") {
            result.heads = positive(required(index, argc, argv), "heads");
        } else if (argument == "--intermediate") {
            result.intermediate =
                positive(required(index, argc, argv), "intermediate");
        } else if (argument == "--patch-size") {
            result.patch_size =
                positive(required(index, argc, argv), "patch size");
        } else if (argument == "--downsample-ratio") {
            result.downsample_ratio =
                positive(required(index, argc, argv), "downsample ratio");
        } else if (argument == "--output-hidden") {
            result.output_hidden =
                positive(required(index, argc, argv), "output hidden");
        } else {
            throw std::invalid_argument(
                "unknown argument: " + argument);
        }
    }
    if (result.model.empty() || result.patches.empty() ||
        result.output.empty()) {
        throw std::invalid_argument(
            "--model, --patches and --output are required");
    }
    return result;
}

std::vector<float> read_floats(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open patch file: " + path);
    }
    input.seekg(0, std::ios::end);
    const auto size = static_cast<std::streamoff>(input.tellg());
    input.seekg(0, std::ios::beg);
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float))) {
        throw std::runtime_error(
            "patch file is not aligned F32 data: " + path);
    }
    std::vector<float> result(
        static_cast<std::size_t>(size / sizeof(float)));
    input.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(size));
    if (!input) {
        throw std::runtime_error("truncated patch file: " + path);
    }
    return result;
}

void write_floats(
    const std::string& path,
    const mfq_tensor_backend::Tensor& value) {
    const auto bytes = static_cast<std::size_t>(value.nbytes());
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path);
    }
    output.write(
        reinterpret_cast<const char*>(value.data_ptr<float>()),
        static_cast<std::streamsize>(bytes));
    if (!output) {
        throw std::runtime_error("cannot write output file: " + path);
    }
}

} // namespace

int main(int argc, char** argv) try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        return 77;
    }
    const auto arguments = parse(argc, argv);
    auto source = mfq::open_model_source(arguments.model);
    CudaExecutionContext execution;
    mfq::cuda::axial_patch_vision::Config config{
        arguments.hidden,
        arguments.layers,
        arguments.heads,
        arguments.intermediate,
        arguments.patch_size,
        arguments.downsample_ratio,
        arguments.output_hidden,
        10'000.0,
        1e-6,
    };
    auto tower =
        mfq::cuda::axial_patch_vision::CudaAxialPatchVisionTower::load(
            execution, *source, config);
    const auto values = read_floats(arguments.patches);
    const auto expected =
        static_cast<std::size_t>(arguments.height) * arguments.width *
        static_cast<std::size_t>(config.patch_width());
    if (values.size() != expected) {
        throw std::invalid_argument(
            "patch file size disagrees with the requested grid");
    }
    auto patches = mfq_tensor_backend::from_blob(
                       const_cast<float*>(values.data()),
                       {static_cast<std::int64_t>(arguments.height) *
                            arguments.width,
                        config.patch_width()},
                       mfq_tensor_backend::TensorOptions()
                           .dtype(mfq_tensor_backend::kFloat32)
                           .device(mfq_tensor_backend::kCPU))
                       .clone()
                       .to(mfq_tensor_backend::kCUDA);
    auto encoded = tower.encode(
        execution, std::move(patches), arguments.height, arguments.width);
    auto host = encoded.to(
                          mfq_tensor_backend::kCPU,
                          mfq_tensor_backend::kFloat32)
                    .contiguous();
    if (host.dim() != 2 || host.size(1) != config.output_hidden) {
        throw std::runtime_error(
            "axial vision output geometry mismatch");
    }
    write_floats(arguments.output, host);
    std::cout << "axial_vision_rows=" << host.size(0)
              << " width=" << host.size(1) << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "DeepSeek-V4.1 axial vision test failed: "
              << error.what() << '\n';
    return 1;
}
