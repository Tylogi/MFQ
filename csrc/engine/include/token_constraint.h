#pragma once

#include "mfq/runtime.h"

struct common_chat_params;

namespace mfq::engine {

class MfqTokenizer;

MfqTokenConstraintPtr make_chat_token_constraint(
    const MfqTokenizer& tokenizer,
    const common_chat_params& params);

} // namespace mfq::engine
