#pragma once

namespace mfq {

// Numeric/layout differences that cannot be represented by a no-copy name
// alias. New schema-v1 artifacts always use canonical semantics; these flags
// exist only while pre-schema files remain readable.
struct TensorLayout {
    double norm_weight_offset = 1.0;
    bool linear_attention_a_is_log = true;
    bool qwen_gdn_gguf_layout = false;
};

} // namespace mfq
