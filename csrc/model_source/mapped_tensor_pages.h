#pragma once
#include <cstddef>

namespace mfq {
// Advise the OS to reclaim clean mapped source pages after weights have
// been copied into their resident tier. Readers retain their mappings and
// can fault pages back in. Returns the number of live views advised.
std::size_t release_mapped_tensor_pages();
}
