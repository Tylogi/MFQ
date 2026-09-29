#pragma once

#include "mfq/runtime.h"

#include <cstddef>

namespace mfq::engine {

// The runtime's only backend-polymorphic boundary.
class Engine {
public:
    Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) noexcept = default;
    Engine& operator=(Engine&&) noexcept = default;
    virtual ~Engine() = 0;

    std::size_t max_concurrent_requests = 1;
    MfqGenerateFn generate;
    MfqReloadFn reload;
    MfqDuplexBackend duplex;
    MfqSessionControl session_control;
    MfqMultimodalGenerateFn multimodal_generate;
    MfqRuntimeMetricsFn runtime_metrics;
};

inline Engine::~Engine() = default;

class CallbackEngine final : public Engine {};

} // namespace mfq::engine

// ponytail: Metal compatibility; remove with its named MetalEngine.
using MfqInferenceEngine = mfq::engine::CallbackEngine;
