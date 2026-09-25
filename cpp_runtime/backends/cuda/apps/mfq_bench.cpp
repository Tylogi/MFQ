#include "qwen_continuous_workload.h"
#include "../benchmarks/qwen_continuous_batching_benchmark.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int64_t number(const std::string& value, const char* option) {
    size_t consumed = 0;
    int64_t result = 0;
    try {
        result = std::stoll(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " requires an integer");
    }
    if (consumed != value.size() || result <= 0) {
        throw std::invalid_argument(std::string(option) + " requires a positive integer");
    }
    return result;
}

void usage() {
    std::cerr << "usage: mfq-bench continuous-batching --model MODEL_PATH "
                 "[--config config.json --ctx-size 16384 "
                 "--prefill-chunk-size 2048 --prefill-tokens 8192 "
                 "--baseline-tokens 8 --gen 16 --reps 5]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) != "continuous-batching") {
            usage();
            return 2;
        }
        std::string model_path;
        std::string config_path;
        int64_t context_size = 16384;
        int64_t chunk_size = 2048;
        int64_t prefill_tokens = 8192;
        int64_t baseline_tokens = 8;
        int64_t generated_tokens = 16;
        int64_t repetitions = 5;
        for (int index = 2; index < argc; ++index) {
            const std::string option(argv[index]);
            if (index + 1 >= argc) {
                throw std::invalid_argument(option + " requires a value");
            }
            const std::string value(argv[++index]);
            if (option == "--model") model_path = value;
            else if (option == "--config") config_path = value;
            else if (option == "--ctx-size") context_size = number(value, option.c_str());
            else if (option == "--prefill-chunk-size") chunk_size = number(value, option.c_str());
            else if (option == "--prefill-tokens") prefill_tokens = number(value, option.c_str());
            else if (option == "--baseline-tokens") baseline_tokens = number(value, option.c_str());
            else if (option == "--gen") generated_tokens = number(value, option.c_str());
            else if (option == "--reps") repetitions = number(value, option.c_str());
            else throw std::invalid_argument("unknown option: " + option);
        }
        if (model_path.empty()) {
            usage();
            return 2;
        }
        if (prefill_tokens >= context_size || context_size <= 17 ||
                generated_tokens > context_size - 17 ||
                prefill_tokens <= chunk_size || baseline_tokens < 2 ||
                repetitions > 100 || generated_tokens > INT32_MAX ||
                baseline_tokens > INT32_MAX || repetitions > INT32_MAX) {
            throw std::invalid_argument("invalid benchmark context, token count or repetitions");
        }
        std::cout << std::unitbuf;
        return mfq::cuda::run_qwen_continuous_workload(
            model_path, config_path, context_size, chunk_size,
            [&](mfq::cuda::QwenContinuousWorkload& workload) {
                return mfq::cuda::continuous::run_qwen_continuous_batching_benchmark(
                    workload, chunk_size, static_cast<int>(generated_tokens),
                    prefill_tokens, static_cast<int>(baseline_tokens),
                    static_cast<int>(repetitions));
            });
    } catch (const std::exception& error) {
        std::cerr << "mfq-bench: " << error.what() << '\n';
        return 1;
    }
}
