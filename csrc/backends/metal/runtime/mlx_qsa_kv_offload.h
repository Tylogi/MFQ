#pragma once

#include "mlx_transformer.h"
#include "qsa_kv_store.h"

#include <memory>
#include <optional>

namespace mfq::metal {

struct MlxQsaKvSnapshot {
    QsaKvSequenceSnapshot rows;
    int heads = 0;
    int dimension = 0;
    int maximum = 0;
    mlx::core::Dtype dtype = mlx::core::float16;
    MlxKvQuantization quantization;
    MlxKvCacheSnapshot materialize(int start, int count) const;
    MlxQsaKvSnapshot prefix(int position) const;
};

struct MlxQsaIndexSnapshot {
    QsaKvSequenceSnapshot rows;
    int width = 0;
    mlx::core::Dtype dtype = mlx::core::float16;
};

class MlxQsaIndexOffload {
public:
    MlxQsaIndexOffload(std::shared_ptr<QsaKvStore> store, int width,
        mlx::core::Dtype dtype, int priority);
    void append(const mlx::core::array& keys);
    mlx::core::array view(int start, int count) const;
    mlx::core::array select(const mlx::core::array& query, int offset, int ratio, int budget) const;
    MlxQsaIndexSnapshot snapshot() const;
    void restore(const MlxQsaIndexSnapshot& state);
    void trim(int position);
    int position() const noexcept;
private:
    QsaKvSequence rows_;
    int width_;
    mlx::core::Dtype dtype_;
};

std::shared_ptr<QsaKvStore> mlx_qsa_kv_offload_store();
void mlx_set_qsa_kv_offload_store(const std::shared_ptr<QsaKvStore>& store);
std::shared_ptr<QsaKvStore> mlx_configure_qsa_kv_offload(
    int maximum, int layers, int heads, int dimension, int index_width, int ratio);

class MlxQsaKvOffload {
public:
    MlxQsaKvOffload(std::shared_ptr<QsaKvStore> store, int heads, int dimension,
        int maximum, int ratio);
    void append(const mlx::core::array& key, const mlx::core::array& value);
    void append_snapshot(const MlxKvCacheSnapshot& source);
    mlx::core::array attention(const mlx::core::array& query,
        const std::optional<mlx::core::array>& blocks, int query_offset, int budget) const;
    MlxQsaKvSnapshot snapshot() const;
    void restore(const MlxQsaKvSnapshot& state);
    void trim(int position);
    int position() const noexcept;
    std::size_t resident_bytes() const noexcept;
private:
    std::shared_ptr<QsaKvStore> store_;
    MlxKvQuantization quantization_;
    QsaKvSequence rows_;
    int heads_, dimension_, maximum_, ratio_;
    mlx::core::Dtype dtype_ = mlx::core::float16;
};

}
