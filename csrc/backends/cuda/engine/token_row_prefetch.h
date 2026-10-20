#pragma once

namespace mfq::cuda::internal {
// Models supply their own token-to-row mapping; generation and MTP share the
// scheduling hook without knowing the embedding format or architecture.
template<class Model,class Ids>
bool prefetch_token_rows(Model& model,const Ids& ids) {
    if constexpr(requires {model.adapter_prefetch_token_rows(model,ids);})
        return model.adapter_prefetch_token_rows(model,ids);
    return false;
}
}
