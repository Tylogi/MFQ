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

struct LoadedCudaEngine {
    MfqInferenceEngine inference;
    CudaEngineMetadata metadata;
};

LoadedCudaEngine load_cuda_engine(CudaEngineOptions options);

int run_cuda_minicpmo_composite(const RuntimeOptions& options);
int run_cuda_minicpmo_duplex(const RuntimeOptions& options);

} // namespace mfq::cuda
