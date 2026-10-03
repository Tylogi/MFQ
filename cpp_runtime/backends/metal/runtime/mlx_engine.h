#pragma once

#include "mlx_generation_job.h"
#include "request_executor.h"

namespace mfq::metal {

// Backend-native resources remain behind the engine, not the communication API.
class MlxEngineModel {
public:
    virtual ~MlxEngineModel() = default;
    virtual void generate(mfq::engine::InferenceRequest&, MlxGenerationJob&) = 0;
    virtual std::int64_t reload(std::int64_t context) = 0;
    virtual mfq::engine::SessionResult session(const mfq::engine::SessionCommand&) = 0;
    virtual mfq::engine::ControlResult control(mfq::engine::ControlRequest) = 0;
};

class MlxEngine final : public mfq::engine::Engine {
public:
    MlxEngine(std::unique_ptr<MlxEngineModel> model,
        std::unique_ptr<mfq::engine::TextProcessor> text,
        mfq::engine::EngineInfo info);
    ~MlxEngine() override;
    mfq::engine::EngineInfo info() const override { return info_; }
    mfq::engine::EngineStatus status() const override;
    mfq::engine::Admission admit(mfq::engine::EngineRequest) override;
    void cancel(const mfq::engine::RequestId&) override;
    mfq::engine::EngineStepResult step(const std::vector<mfq::engine::RequestId>&) override;
    mfq::engine::SessionResult session(const mfq::engine::SessionCommand&) override;
    mfq::engine::ControlResult control(mfq::engine::ControlRequest) override;
    std::int64_t reload(std::int64_t context) override;
    void shutdown() override;

    // RequestExecutor operations: Metal's native pipeline is serial, but does
    // not block the scheduler on device execution or a slow response consumer.
    bool can_batch(const mfq::engine::EngineRequest&) const { return false; }
    bool exclusive() const { return duplex_active_; }
    bool mtp_available() const { return info_.capabilities.mtp; }
    void execute(const std::vector<mfq::engine::RequestId>&) {}
    mfq::engine::Generation generate(const mfq::engine::RequestId&,
        mfq::engine::ExecutionRequest& request);

private:
    std::unique_ptr<MlxEngineModel> model_;
    std::unique_ptr<mfq::engine::TextProcessor> text_;
    mfq::engine::EngineInfo info_;
    mfq::engine::RequestExecutor requests_;
    bool duplex_active_ = false;
};

} // namespace mfq::metal
