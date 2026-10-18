#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::models {

// Media encoders emit rows in prompt order; position policies are applied by
// the caller after replacing placeholders in the text embedding sequence.
template<class Tensor, class Embed, class Scatter>
auto inject_vision_embeddings(const std::vector<int64_t>& ids, Tensor merged,
        int64_t count, int64_t image, int64_t video, bool accept_video,
        Embed embed, Scatter scatter) {
    std::vector<int64_t> placeholders;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == image || (ids[i] == video && accept_video))
            placeholders.push_back(static_cast<int64_t>(i));
        else if (ids[i] == video)
            throw std::invalid_argument("image preparation does not accept video placeholders");
    }
    if (static_cast<int64_t>(placeholders.size()) != count)
        throw std::invalid_argument("image/video placeholders disagree with merged vision patches");
    auto embeddings = embed(ids);
    scatter(embeddings, merged, placeholders);
    return embeddings;
}

} // namespace mfq::models
