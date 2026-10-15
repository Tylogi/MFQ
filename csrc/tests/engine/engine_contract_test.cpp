#include "engine.h"
#include "mtp_metrics.h"
#include <type_traits>

#if __has_include("inference.h") || __has_include("text_processor.h") || \
    __has_include("request_table.h") || __has_include("generation_step.h") || \
    __has_include("tokenizer.h")
#error Engine implementation leaked through the public include surface
#endif

static_assert(std::is_move_constructible_v<mfq::engine::EngineRequest>);
static_assert(std::is_copy_constructible_v<mfq::engine::EngineEvent>);

int main() {
    mfq::engine::EngineRequest request;
    request.input.prompt = "owned text";
    return request.diagnostic_input.has_value();
}
