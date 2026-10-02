#pragma once

#include "cuda_runtime_config.h"
#include "engine.h"

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

struct CudaEngine final : mfq::engine::Engine {
    struct Impl;
    explicit CudaEngine(std::unique_ptr<Impl> impl);
    CudaEngine(CudaEngine&&) noexcept;
    CudaEngine& operator=(CudaEngine&&) noexcept;
    ~CudaEngine() override;
    mfq::engine::EngineInfo info() const override;
    mfq::engine::Admission admit(mfq::engine::EngineRequest request) override;
    void cancel(const mfq::engine::RequestId& id) override;
    mfq::engine::EngineStepResult step(const std::vector<mfq::engine::RequestId>& eligible) override;
    mfq::engine::EngineStatus status() const override;
    mfq::engine::SessionResult session(const mfq::engine::SessionCommand& command) override;
    std::int64_t reload(std::int64_t context) override;
    void shutdown() override;
    mfq::engine::ControlResult control(mfq::engine::ControlRequest request) override;
    const CudaEngineMetadata& metadata() const;
private:
    friend CudaEngine load_cuda_engine(CudaEngineOptions options);
    std::unique_ptr<Impl> impl_;
};

CudaEngine load_cuda_engine(CudaEngineOptions options);

} // namespace mfq::cuda
