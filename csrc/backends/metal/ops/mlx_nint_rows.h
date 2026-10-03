#pragma once

#include "mfq/nint_rows.h"
#include <span>
#include <mlx/mlx.h>

namespace mfq::metal {
class MlxMappedNintRows : public mfq::NintRows {
public:
    explicit MlxMappedNintRows(std::span<const std::uint8_t> blob)
        : NintRows(blob.data(), blob.size()) {}
};

class MlxNintRowBatch : public mfq::NintRowBatch {
public:
    mlx::core::array decode(mlx::core::Dtype dtype = mlx::core::float16) const;
};
} // namespace mfq::metal
