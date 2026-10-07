#include "mlx_inference_warmup.h"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    try {
        mfq::metal::MlxSamplingParams defaults;
        defaults.temperature = 0.8;
        defaults.top_k = 64;
        int calls = 0, resets = 0, draft_mask = 0;
        mfq::metal::warm_mlx_inference({1, 2, 3}, 128, 64, defaults, true,
            [&](const auto& prompt, const auto& sampling, int count) {
                require(prompt.size() + count - 1 <= 128, "warmup exceeded context");
                if (calls == 0) {
                    require(prompt.size() == 64 && count == 2, "bulk prefill not covered");
                    require(prompt[3] == 1, "bulk prompt not repeated safely");
                }
                if (calls == 1) require(prompt.size() == 63 && count == 2, "unaligned prefill not covered");
                if (sampling.enable_mtp) {
                    draft_mask |= 1 << sampling.mtp_max_draft_tokens;
                    require(count == sampling.mtp_max_draft_tokens + 2, "verification depth not covered");
                }
                if (calls == 13) require(!sampling.enable_mtp && sampling.greedy(), "last pass not ordinary greedy");
                ++calls;
            }, [&] { ++resets; });
        require(calls == 14 && resets == 1 && draft_mask == 62, "warmup coverage mismatch");
        calls = resets = 0;
        mfq::metal::warm_mlx_inference({1, 2, 3}, 2, 64, defaults, true,
            [&](const auto& prompt, const auto& sampling, int count) {
                require(prompt.size() == 1 && count == 2 && !sampling.enable_mtp,
                    "small context warmup is unsafe");
                ++calls;
            }, [&] { ++resets; });
        require(calls == 2 && resets == 1, "small context coverage mismatch");
        calls = resets = 0;
        try {
            mfq::metal::warm_mlx_inference({1}, 128, 64, defaults, false,
                [&](const auto&, const auto&, int) { ++calls; throw std::runtime_error("generation failed"); },
                [&] { ++resets; throw std::runtime_error("cleanup failed"); });
            throw std::logic_error("warmup swallowed failure");
        } catch (const std::runtime_error& error) {
            require(std::string(error.what()) == "generation failed", "cleanup masked the generation failure");
        }
        require(calls == 1 && resets == 1, "failed warmup did not clean up");
        std::cout << "Inference warmup coverage and cleanup passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
