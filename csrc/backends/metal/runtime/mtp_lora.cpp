#include "mtp_lora.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace mfq::metal {
namespace {
void geometry(int inputs, int outputs, int rank) {
    if (inputs <= 0 || inputs > 8388608 || outputs <= 0 || outputs > 8388608 || rank <= 0 || rank > 32)
        throw std::invalid_argument("invalid packed MTP LoRA geometry");
}
}

MtpLoraState MtpLoraState::create(int inputs, int outputs, int rank, bool randomize) {
    geometry(inputs, outputs, rank);
    MtpLoraState result;
    result.inputs = inputs;
    result.outputs = outputs;
    result.rank = rank;
    result.a.resize(std::size_t(inputs) * rank);
    result.b.resize(std::size_t(outputs) * rank);
    result.first_moment.resize(result.a.size() + result.b.size());
    result.second_moment.resize(result.first_moment.size());
    std::mt19937 generator(0x4d465154);
    std::normal_distribution<float> normal(0, 1.0f / std::sqrt(float(inputs)));
    if (randomize) for (auto& value : result.a) value = normal(generator);
    return result;
}

void MtpLoraState::validate() const {
    geometry(inputs, outputs, rank);
    if (a.size() != std::size_t(inputs) * rank || b.size() != std::size_t(outputs) * rank ||
        first_moment.size() != a.size() + b.size() || second_moment.size() != first_moment.size() ||
        version != optimizer_step || optimizer_step > 1000000000)
        throw std::invalid_argument("invalid MTP LoRA state shape or version");
    for (const auto* values : {&a, &b, &first_moment, &second_moment}) for (auto value : *values)
        if (!std::isfinite(value) || std::abs(value) > 65504)
            throw std::invalid_argument("non-finite MTP LoRA state");
    for (auto value : second_moment) if (value < 0)
        throw std::invalid_argument("negative MTP LoRA second moment");
}

std::size_t MtpLoraState::bytes() const noexcept {
    return (a.size() + b.size() + first_moment.size() + second_moment.size()) * sizeof(float);
}

std::shared_ptr<MtpLoraState> mtp_lora_adam(const MtpLoraState& state, std::span<const float> gradients) {
    state.validate();
    if (gradients.size() != state.a.size() + state.b.size())
        throw std::invalid_argument("MTP LoRA gradient shape mismatch");
    double squared_norm = 0;
    for (auto value : gradients) {
        if (!std::isfinite(value)) return {};
        squared_norm += double(value) * value;
    }
    const auto clipping = 1.0 / std::max(1.0, std::sqrt(squared_norm));
    auto candidate = std::make_shared<MtpLoraState>(state);
    ++candidate->optimizer_step;
    ++candidate->version;
    const auto beta1 = 1 - std::pow(0.9, double(candidate->optimizer_step));
    const auto beta2 = 1 - std::pow(0.999, double(candidate->optimizer_step));
    for (std::size_t i = 0; i < gradients.size(); ++i) {
        const auto g = gradients[i] * clipping;
        auto& first = candidate->first_moment[i];
        auto& second = candidate->second_moment[i];
        first = 0.9 * first + 0.1 * g;
        second = 0.999 * second + 0.001 * g * g;
        auto& value = i < state.a.size() ? candidate->a[i] : candidate->b[i - state.a.size()];
        value -= 0.00005f * (first / beta1) / (std::sqrt(second / beta2) + 1e-8);
    }
    candidate->validate();
    return candidate;
}

} // namespace mfq::metal
