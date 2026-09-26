#pragma once

#include "options.h"
#include "mfq/runtime.h"

#include <cstdint>
#include <memory>
#include <string>

namespace mfq {
class ModelSource;
}

namespace mfq::cuda {

struct CudaEngineCapabilities {
    bool text = false;
    bool image_input = false;
    bool video_input = false;
    bool audio_input = false;
    bool audio_output = false;
    bool full_duplex = false;
    bool mtp = false;
};

struct CudaEngineMetadata {
    std::shared_ptr<const mfq::ModelSource> source;
    std::string architecture;
    std::string model_type;
    std::int64_t max_context = 0;
    std::int64_t vocab_size = 0;
    CudaEngineCapabilities capabilities;
};

struct CudaInferenceEngine final : MfqInferenceEngine {
    CudaEngineMetadata metadata;
};

CudaInferenceEngine load_cuda_engine(CudaEngineOptions options);

} // namespace mfq::cuda
