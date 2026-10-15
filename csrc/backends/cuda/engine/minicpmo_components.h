#pragma once

#include "inference.h"
#include "models/minicpmo45/ops.h"

#include <memory>

namespace mfq::cuda::minicpmo45 {

class Components {
  public:
    explicit Components(MiniCPMO45CausalLm language);
    ~Components();

    // The composite replaces the loaded language model; the engine binds this
    // through the family-independent interface when it composes its state.
    CudaCausalModel& language() noexcept;
    mfq::StepSequence<CudaPreparedPrompt> prepare(
        const std::vector<int64_t>& prompt, const MfqMultimodalInput& media);
    mfq::StepSequence<std::monostate> start(MfqDuplexSessionParams parameters);
    mfq::StepSequence<MfqDuplexStepResult> step(MfqDuplexStepInput input);
    void stop();
    std::size_t state_memory_bytes() const;

  private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace mfq::cuda::minicpmo45
