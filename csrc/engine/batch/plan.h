#pragma once

#include "policy.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace mfq::engine {

// What one physical step will run: the decode rows, or a single prefill
// chunk. Never both.
template <class Request>
struct BatchPlan {
    std::vector<std::shared_ptr<Request>> decode;
    std::shared_ptr<Request> prefill;
    PrefillChunk prefill_chunk;
};

// BatchPlanner is this function, not an object. It reads the facts the
// request rows already carry (active rows, pending prefill chunk, whether a
// decode row was requested) plus the scheduler's eligible set and the
// decode/prefill preference, writes nothing, and the caller executes the
// result. Admission, fairness and MTP depth stay where they are decided.
//
//   active   - rows that have completed prefill and hold execution state
//   pending  - per request work record: .request, .prefill, .decode
//   eligible - request IDs the scheduler allows to run this step
template <class Active, class Pending, class Eligible>
BatchPlan<typename Active::value_type::element_type> plan_batch(
        const Active &active, const Pending &pending, const Eligible &eligible,
        bool prefer_decode) {
    BatchPlan<typename Active::value_type::element_type> plan;
    for (const auto &request : active) {
        const auto &work = *pending.at(request->id);
        if (work.decode && std::find(eligible.begin(), eligible.end(), request->id) != eligible.end())
            plan.decode.push_back(request);
    }
    for (const auto &id : eligible) {
        const auto it = pending.find(id);
        if (it == pending.end() || !it->second->prefill) continue;
        plan.prefill = it->second->request;
        plan.prefill_chunk = *it->second->prefill;
        break;
    }
    if (plan.decode.empty() || (plan.prefill && !prefer_decode)) plan.decode.clear();
    else plan.prefill.reset();
    return plan;
}

} // namespace mfq::engine
