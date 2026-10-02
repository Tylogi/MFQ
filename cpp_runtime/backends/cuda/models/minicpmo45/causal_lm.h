#pragma once

#include "runtime.h"

#include <mutex>
#include <optional>

namespace mfq::cuda::minicpmo45 {

class Components {
public:
    explicit Components(mfq::cuda::MiniCPMO45CausalLm language);

    mfq::cuda::MiniCPMO45CausalLm& language() noexcept;
    MfqMultimodalGenerateFn multimodal_generate(std::mutex& model_mutex);
    MfqDuplexBackend duplex(std::mutex& model_mutex);

private:
    MiniCPMO45Runtime runtime_;
    std::optional<MiniCPMO45DuplexSession> duplex_session_;
};

} // namespace mfq::cuda::minicpmo45
