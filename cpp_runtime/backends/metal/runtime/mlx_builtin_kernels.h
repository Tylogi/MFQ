#pragma once

#include <string>

namespace mfq::metal {

std::string qsa_prefill_indexer_source();
std::string sparse_attention_source();
std::string sdpa_decode_source();
std::string dsv4_mxfp4_blocks_source();

}
