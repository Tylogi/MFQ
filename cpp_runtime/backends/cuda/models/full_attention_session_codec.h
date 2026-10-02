#pragma once

#include "session_state.h"

namespace mfq::cuda {

template <typename Model> struct CudaCausalOps;
template <typename Model>
using CausalLm = typename Model::template CausalModel<CudaCausalOps<Model>>;

template <typename Model> struct FullAttentionSessionCodec {
    using CausalModel = CausalLm<Model>;
    static TextSessionStateKind kind(const CausalModel &model);
    static bool supports_paged(const CausalModel &model);
    static TextSessionState capture(const CausalModel &model, const std::vector<int64_t> &tokens);
    static void restore(CausalModel &model, const TextSessionState &state);
};

} // namespace mfq::cuda
