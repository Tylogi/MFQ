#pragma once

#include <mlx/mlx.h>

namespace mfq::metal {

// Drain both the explicitly shared runtime stream and the calling thread's
// default stream before releasing arrays or reclaiming MLX allocator pages.
// MTP and sampling use async_eval, so holding the runtime mutex alone does not
// prove that Metal command buffers have stopped referencing their storage.
inline void drain_metal_work(const mlx::core::Stream& runtime_stream) {
    mlx::core::synchronize(runtime_stream);
    mlx::core::synchronize();
}

} // namespace mfq::metal
