#include "mlx_multimodal.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace mfq::metal {

mlx::core::array replace_multimodal_embedding_span(
    const mlx::core::array& embeddings,
    const mlx::core::array& replacement,
    int batch,
    int begin,
    int end) {
    if (embeddings.ndim() != 3 || replacement.ndim() != 3 ||
        batch < 0 || batch >= embeddings.shape(0) || begin < 0 ||
        end <= begin || end > embeddings.shape(1) ||
        replacement.shape(0) != 1 ||
        replacement.shape(1) != end - begin ||
        replacement.shape(2) != embeddings.shape(2)) {
        throw std::invalid_argument(
            "multimodal embedding replacement span is incompatible");
    }
    auto value = replacement;
    if (value.dtype() != embeddings.dtype()) {
        value = mlx::core::astype(value, embeddings.dtype());
    }
    return mlx::core::slice_update(
        embeddings,
        value,
        mlx::core::Shape{batch, begin, 0},
        mlx::core::Shape{
            batch + 1, end, embeddings.shape(2)});
}

mlx::core::array replace_multimodal_token_embeddings(
    const mlx::core::array& embeddings,
    const mlx::core::array& replacement,
    const std::vector<std::int64_t>& token_ids,
    const std::vector<std::int64_t>& placeholder_ids) {
    if (embeddings.ndim() != 3 || embeddings.shape(0) != 1 ||
        embeddings.shape(1) != static_cast<int>(token_ids.size()) ||
        replacement.ndim() != 2 ||
        replacement.shape(1) != embeddings.shape(2) ||
        placeholder_ids.empty()) {
        throw std::invalid_argument(
            "multimodal token-embedding geometry is incompatible");
    }
    const std::unordered_set<std::int64_t> placeholders(
        placeholder_ids.begin(), placeholder_ids.end());
    auto output = embeddings;
    int consumed = 0;
    std::size_t begin = 0;
    while (begin < token_ids.size()) {
        if (!placeholders.contains(token_ids[begin])) {
            ++begin;
            continue;
        }
        std::size_t end = begin + 1;
        while (end < token_ids.size() &&
               placeholders.contains(token_ids[end])) {
            ++end;
        }
        const int count = static_cast<int>(end - begin);
        if (count > replacement.shape(0) - consumed) {
            throw std::invalid_argument(
                "multimodal placeholders exceed replacement embeddings");
        }
        auto values = mlx::core::reshape(
            mlx::core::slice(
                replacement,
                mlx::core::Shape{consumed, 0},
                mlx::core::Shape{consumed + count, replacement.shape(1)}),
            mlx::core::Shape{1, count, replacement.shape(1)});
        output = replace_multimodal_embedding_span(
            output,
            values,
            0,
            static_cast<int>(begin),
            static_cast<int>(end));
        consumed += count;
        begin = end;
    }
    if (consumed != replacement.shape(0)) {
        throw std::invalid_argument(
            "replacement embeddings exceed multimodal placeholders");
    }
    return output;
}

MlxGridMropePositions build_grid_mrope_positions(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t image_token_id,
    std::int64_t video_token_id,
    int spatial_merge_size,
    const std::vector<MlxGridShape>& image_grids,
    const std::vector<MlxGridShape>& video_grids) {
    auto positions = ::mfq::build_grid_mrope_positions(
        token_ids, image_token_id, video_token_id, spatial_merge_size,
        image_grids, video_grids);
    return {
        mlx::core::array(
            positions.values.begin(),
            mlx::core::Shape{3, static_cast<int>(positions.token_count)},
            mlx::core::int32),
        positions.decode_delta,
    };
}

} // namespace mfq::metal
