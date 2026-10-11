#include "mlx_inference_warmup.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace mfq::metal {

namespace {
void warm_sorting() {
    for (const auto dtype : {mlx::core::int32, mlx::core::float32}) {
        for (const int size : {128, 256, 512, 1024, 2048, 2049}) {
            const auto input = mlx::core::zeros({size}, dtype, mlx::core::Device::gpu);
            mlx::core::eval(mlx::core::argsort(input, -1, mlx::core::Device::gpu));
            if (dtype == mlx::core::int32)
                mlx::core::eval(mlx::core::sort(input, -1, mlx::core::Device::gpu));
        }
    }
}
}

void warm_mlx_inference(const std::vector<std::int64_t>& prompt,
    int context, int chunk_size, const MlxSamplingParams& defaults, bool mtp,
    const MlxWarmupGenerate& generate, const std::function<void()>& reset) {
    if (prompt.empty() || context < 1 || chunk_size < 1)
        throw std::invalid_argument("invalid inference warmup configuration");
    const auto started = std::chrono::steady_clock::now();
    const int rows = std::min({32, chunk_size, static_cast<int>(prompt.size()), std::max(1, context - 8)});
    const std::vector<std::int64_t> short_prompt(prompt.begin(), prompt.begin() + rows);
    const int limit = std::min(8, context - rows + 1);
    const int depth = mtp ? std::max(0, std::min(5, limit - 2)) : 0;
    const int bulk_rows = std::min(chunk_size, std::max(1, context - 8));
    const bool bulk = bulk_rows > rows;
    const int tail_rows = std::min(65, bulk_rows - 1);
    const bool tail = tail_rows > rows;
    const int total = 2 + 2 * depth + (bulk ? 1 : 0) + (tail ? 1 : 0);
    int completed = 0;
    const auto progress = [&] {
        std::cerr << "mfq_load_progress stage=warming completed=" << completed
                  << " total=" << total << std::endl;
    };
    const auto run = [&](const auto& tokens, const auto& sampling, int count) {
        generate(tokens, sampling, count);
        ++completed;
        progress();
    };
    progress();
    try {
        warm_sorting();
        MlxSamplingParams greedy;
        greedy.enable_mtp = false;
        if (bulk) {
            std::vector<std::int64_t> tokens;
            tokens.reserve(bulk_rows);
            for (int row = 0; row < bulk_rows; ++row)
                tokens.push_back(prompt[row % prompt.size()]);
            run(tokens, greedy, 2);
        }
        if (tail) {
            std::vector<std::int64_t> tokens;
            tokens.reserve(tail_rows);
            for (int row = 0; row < tail_rows; ++row)
                tokens.push_back(prompt[row % prompt.size()]);
            run(tokens, greedy, 2);
        }
        auto sampled = defaults;
        sampled.enable_mtp = false;
        for (auto sampling : {greedy, sampled}) {
            for (int draft = 1; draft <= depth; ++draft) {
                sampling.enable_mtp = true;
                sampling.mtp_max_draft_tokens = draft;
                run(short_prompt, sampling, draft + 2);
            }
        }
        run(short_prompt, sampled, limit);
        run(short_prompt, greedy, limit);
    } catch (...) {
        try { reset(); } catch (...) {}
        throw;
    }
    reset();
    std::cout << "inference_warmup passes=" << completed << " seconds="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
              << std::endl;
}

}
