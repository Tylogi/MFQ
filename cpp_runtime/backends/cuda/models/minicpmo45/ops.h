#pragma once

#include "runtime.h"

#include <optional>

namespace mfq::cuda::minicpmo45 {

class Components {
  public:
    explicit Components(mfq::cuda::MiniCPMO45CausalLm language);

    mfq::cuda::MiniCPMO45CausalLm &language() noexcept;
    CudaPreparedPrompt prepare(const std::vector<int64_t> &prompt, const MfqMultimodalInput &media);
    void start(const MfqDuplexSessionParams &parameters);
    MfqDuplexStepResult step(const MfqDuplexStepInput &input);
    void stop();

  private:
    MiniCPMO45Runtime runtime_;
    std::optional<MiniCPMO45DuplexSession> duplex_session_;
};

} // namespace mfq::cuda::minicpmo45
