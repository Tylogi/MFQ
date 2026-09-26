#pragma once

#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>

namespace mfq::engine {

// Ops supplies:
//   using Tensor;
//   using Params;
//   Tensor sample_greedy(Tensor);
//   Tensor sample_stochastic(Tensor, float random, const Params&);
//   Tensor apply_penalties(Tensor, const Tensor& counts, const Params&);
// Tensor arguments are consumed, so an Ops implementation may update them in
// place. Callers that still need the original tensor must clone it first.
template <typename Ops>
class Sampler {
public:
    using Tensor = typename Ops::Tensor;
    using Params = typename Ops::Params;

    explicit Sampler(Params params = Params{}, Ops ops = Ops{})
        : params_(std::move(params)),
          ops_(std::move(ops)),
          rng_(params_.seed) {
        if (params_.top_k < 0) {
            throw std::invalid_argument("top_k must be non-negative");
        }
    }

    const Params& params() const noexcept {
        return params_;
    }

    Ops& ops() noexcept {
        return ops_;
    }

    const Ops& ops() const noexcept {
        return ops_;
    }

    bool greedy() const noexcept {
        return params_.temperature <= 0.0 || params_.top_k == 1;
    }

    bool has_penalties() const noexcept {
        return params_.presence_penalty != 0.0 ||
            params_.frequency_penalty != 0.0 ||
            params_.repetition_penalty != 1.0;
    }

    void reset_seed(std::uint64_t seed) {
        params_.seed = seed;
        rng_.seed(seed);
    }

    void discard_random(std::uint64_t draws) {
        rng_.discard(draws);
    }

    float next_uniform_float() {
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        return uniform(rng_);
    }

    Tensor sample(Tensor logits) {
        if (greedy()) {
            return ops_.sample_greedy(std::move(logits));
        }
        return ops_.sample_stochastic(
            std::move(logits), next_uniform_float(), params_);
    }

    Tensor sample(Tensor logits, const Tensor& counts) {
        if (has_penalties()) {
            logits = apply_penalties(std::move(logits), counts);
        }
        return sample(std::move(logits));
    }

    Tensor apply_penalties(Tensor logits, const Tensor& counts) {
        return ops_.apply_penalties(
            std::move(logits), counts, params_);
    }

    double next_uniform() {
        return std::generate_canonical<double, 53>(rng_);
    }

private:
    Params params_;
    Ops ops_;
    std::mt19937_64 rng_;
};

} // namespace mfq::engine
