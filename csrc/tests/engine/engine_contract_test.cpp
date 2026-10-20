#include "engine.h"
#include "mtp_metrics.h"
#include "step_sequence.h"
#include <type_traits>

#if __has_include("inference.h") || __has_include("text_processor.h") || \
    __has_include("requests.h") || __has_include("generation_step.h") || \
    __has_include("tokenizer.h") || __has_include("qwen35/config.h")
#error Engine implementation leaked through the public include surface
#endif

static_assert(std::is_same_v<decltype(mfq::engine::InferenceInput{}.sampling), mfq::SamplingParams>);
static_assert(std::is_move_constructible_v<mfq::StepSequence<int>>);
static_assert(std::is_move_constructible_v<mfq::engine::EngineRequest>);
static_assert(std::is_copy_constructible_v<mfq::engine::EngineEvent>);

int main() {
    mfq::engine::EngineRequest request;
    request.input.prompt = "owned text";
    return request.diagnostic_input.has_value();
}
