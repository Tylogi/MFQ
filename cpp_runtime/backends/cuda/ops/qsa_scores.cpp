#include "qsa_scores.h"
#include "mfq/kernels/cuda/qsa_scores.h"

namespace mfq::cuda {
Tensor qsa_block_scores(const Tensor& query, const Tensor& pooled_keys) {
    return mfq_qsa::block_scores(query, pooled_keys);
}
} // namespace mfq::cuda
