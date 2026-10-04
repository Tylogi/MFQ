#include "inference.h"
#include "generation_policy.h"
#include <cassert>
int main() {
    using namespace mfq::engine;
    InferenceMetrics metrics;
    metrics.mark_token();
    assert(metrics.saw_token && metrics.first_token == metrics.last_token);
    const auto first = metrics.first_token;
    metrics.mark_token();
    assert(metrics.first_token == first && metrics.last_token >= first);
    InferenceRequest request; request.prompt = {1, 2, 3}; request.sampling.max_tokens = 3;
    auto plan = plan_generation(request.prompt, 32, 8, 20, 2);
    assert(plan.generation_tokens == 5 && plan.stable_prefix_tokens == 2);
    auto chunk = next_prefill_chunk(9, 4, 3);
    assert(chunk.count == 3 && !chunk.last);
    InferenceOutput output(request, nullptr, "raw");
    auto delta = output.append({4, 5, 6, 7});
    assert((delta.token_ids == std::vector<int64_t>{4, 5, 6}));
    assert(output.stopped() && output.result.completion_tokens == 3);
    assert(output.metrics.saw_token && output.metrics.last_token >= output.metrics.first_token);
    const auto last = output.metrics.last_token;
    assert(output.append({8}).token_ids.empty());
    assert(output.metrics.last_token == last);
    output.result.cancelled = true;
    output.finish();
    assert(output.result.finish_reason == "cancelled");
    assert(output.metrics.last_token == last);
    assert(output.finish().diffs.empty());
}
