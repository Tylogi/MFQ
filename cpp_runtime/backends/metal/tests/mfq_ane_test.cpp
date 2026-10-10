#include "mfq_ane.h"
#include "mtp_lora.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc, char**) {
    try {
        mfq::metal::AneMatmul multiply(32, 64, 32);
        std::vector<float> a(32 * 64), b(64 * 32);
        for (std::size_t i = 0; i < a.size(); ++i) a[i] = std::sin(float(i)) * 0.1f;
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::cos(float(i)) * 0.1f;
        for (int iteration = 0; iteration < 2; ++iteration) {
            const auto result = multiply(a, b);
            for (int m = 0; m < 32; ++m) for (int n = 0; n < 32; ++n) {
                float expected = 0;
                for (int k = 0; k < 64; ++k) expected += a[m * 64 + k] * b[k * 32 + n];
                if (!std::isfinite(result[m * 32 + n]) || std::abs(result[m * 32 + n] - expected) > 0.002f)
                    throw std::runtime_error("ANE dynamic matmul differs from reference");
            }
            for (auto& value : b) value *= -0.5f;
        }
        std::cout << "ANE dynamic weights: two evaluations passed\n";
        auto state = mfq::metal::MtpLoraState::create(64, 65, 16);
        std::vector<float> gradient(state.a.size() + state.b.size(), 0.01f);
        const auto update = mfq::metal::mtp_lora_adam(state, gradient);
        if (!update || update->version != 1 || update->a == state.a || state.version != 0)
            throw std::runtime_error("LoRA optimizer mutated its immutable input");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
