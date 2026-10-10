#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace mfq::metal {

struct MtpLoraState {
    int inputs = 0, outputs = 0, rank = 0;
    std::uint64_t version = 0, optimizer_step = 0;
    std::vector<float> a, b, first_moment, second_moment;

    static MtpLoraState create(int inputs, int outputs, int rank = 16, bool randomize = true);
    void validate() const;
    std::size_t bytes() const noexcept;
};

struct MtpLoraExample {
    virtual ~MtpLoraExample() = default;
    virtual std::size_t bytes() const noexcept = 0;
};

struct MtpLoraBatch {
    int rows = 0;
    std::vector<std::shared_ptr<const MtpLoraExample>> examples;
    std::vector<float> teacher_logits, position_weights;
    std::size_t workspace_bytes = 0;
};

struct MtpLoraUpdate {
    std::shared_ptr<const MtpLoraState> state;
    double loss_before = 0, loss_after = 0, trust_kl = 0;
};

std::shared_ptr<MtpLoraState> mtp_lora_adam(const MtpLoraState&, std::span<const float> gradients);

} // namespace mfq::metal
