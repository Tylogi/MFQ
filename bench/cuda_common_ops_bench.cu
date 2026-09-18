#include "cuda_common_ops/common.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mfq::bench {
namespace {

std::int64_t parse_int64(const char* text, const char* option) {
    std::size_t parsed = 0;
    const std::string value(text);
    std::int64_t result = 0;
    try {
        result = std::stoll(value, &parsed);
    } catch (const std::exception&) {
        usage_error(std::string(option) + " expects an integer");
    }
    if (parsed != value.size()) {
        usage_error(std::string(option) + " expects an integer");
    }
    return result;
}

double parse_double(const char* text, const char* option) {
    std::size_t parsed = 0;
    const std::string value(text);
    double result = 0.0;
    try {
        result = std::stod(value, &parsed);
    } catch (const std::exception&) {
        usage_error(std::string(option) + " expects a number");
    }
    if (parsed != value.size() || !std::isfinite(result)) {
        usage_error(std::string(option) + " expects a finite number");
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) usage_error("missing operation");
    const std::string_view first(argv[1]);
    if (first == "--help" || first == "-h") {
        std::cout
            << "usage: mfq-cuda-common-ops-bench "
               "<acc|acc-rms-norm|silu-mul|rms-norm|rope> [options]\n";
        std::exit(0);
    }

    Options options;
    options.operation = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string_view option(argv[index]);
        const auto value = [&]() -> const char* {
            if (index + 1 >= argc) {
                usage_error(std::string(option) + " requires a value");
            }
            return argv[++index];
        };
        if (option == "--rows") {
            options.rows = parse_int64(value(), "--rows");
        } else if (option == "--width") {
            options.width = parse_int64(value(), "--width");
        } else if (option == "--rotary-width") {
            options.rotary_width = parse_int64(value(), "--rotary-width");
        } else if (option == "--dtype") {
            options.dtype = value();
        } else if (option == "--warmup") {
            options.warmup = static_cast<int>(parse_int64(value(), "--warmup"));
        } else if (option == "--iterations") {
            options.iterations = static_cast<int>(
                parse_int64(value(), "--iterations"));
        } else if (option == "--samples") {
            options.samples = static_cast<int>(parse_int64(value(), "--samples"));
        } else if (option == "--seed") {
            options.seed = parse_int64(value(), "--seed");
        } else if (option == "--epsilon") {
            options.epsilon = parse_double(value(), "--epsilon");
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }

    if (options.operation != "acc" && options.operation != "acc-rms-norm" &&
        options.operation != "silu-mul" && options.operation != "rms-norm" &&
        options.operation != "rope") {
        usage_error("unsupported operation: " + options.operation);
    }
    if (options.dtype != "f16" && options.dtype != "bf16" &&
        options.dtype != "f32") {
        usage_error("--dtype must be f16, bf16, or f32");
    }
    if (options.rows <= 0 || options.width <= 0) {
        usage_error("--rows and --width must be positive");
    }
    if (options.rotary_width < 0) {
        usage_error("--rotary-width must be non-negative");
    }
    if (options.rows > std::numeric_limits<int>::max() / options.width) {
        usage_error("rows * width exceeds the CUDA operator index range");
    }
    if (options.warmup < 0 || options.iterations <= 0 || options.samples <= 0) {
        usage_error(
            "warmup must be non-negative; iterations and samples must be positive");
    }
    if (!(options.epsilon > 0.0)) {
        usage_error("--epsilon must be positive");
    }
    return options;
}

}  // namespace
}  // namespace mfq::bench

int main(int argc, char** argv) {
    try {
        const auto options = mfq::bench::parse_options(argc, argv);
        int devices = 0;
        const auto count_status = cudaGetDeviceCount(&devices);
        if (count_status != cudaSuccess || devices == 0) {
            (void)cudaGetLastError();
            return 77;
        }
        cudaDeviceProp properties{};
        mfq::bench::check_cuda(
            cudaGetDeviceProperties(&properties, 0),
            "cudaGetDeviceProperties");
        const mfq::cuda::Device gpu{mfq::cuda::DeviceType::cuda, 0};
        if (options.operation == "acc") {
            mfq::bench::run_acc(options, gpu, properties);
        } else if (options.operation == "acc-rms-norm") {
            mfq::bench::run_acc_rms_norm(options, gpu, properties);
        } else if (options.operation == "silu-mul") {
            mfq::bench::run_silu_mul(options, gpu, properties);
        } else if (options.operation == "rms-norm") {
            mfq::bench::run_rms_norm(options, gpu, properties);
        } else {
            mfq::bench::run_rope(options, gpu, properties);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA common-ops benchmark failed: " << error.what()
                  << '\n';
        return 1;
    }
}
