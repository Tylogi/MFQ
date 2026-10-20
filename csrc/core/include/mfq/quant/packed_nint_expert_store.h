#pragma once

#include "mfq/quant/mfe_expert_store.h"

namespace mfq {
// Present packed [experts, out, in] NINT as a virtual MFE container. Only the
// small shape/ownership header is rewritten; compressed payloads stay in place.
std::shared_ptr<MfeQuantExpertStore> make_packed_nint_expert_store(
    std::size_t bytes, MfeQuantExpertStore::Read read,NintRows::ReadBatch batch={});
}
